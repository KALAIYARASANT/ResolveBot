"""Real HTTP return-clarification retest with temporary SQLite writes."""
from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUT_RATE = 0.10
CACHED_RATE = 0.01
OUTPUT_RATE = 0.50

import uvicorn
from dateutil.relativedelta import relativedelta
from langchain_core.callbacks import BaseCallbackHandler

from app.core.config import get_settings
import app.agents.support_agent as support_agent


class UsageCapture(BaseCallbackHandler):
    def __init__(self):
        self.active_query = "unassigned"
        self.started = {}
        self.calls = []

    def _start(self, serialized, run_id):
        kwargs = (serialized or {}).get("kwargs", {})
        self.started[str(run_id)] = {
            "query": self.active_query,
            "start": time.perf_counter(),
            "model": kwargs.get("model") or kwargs.get("model_name"),
        }

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(serialized, run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(serialized, run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        record = self.started.pop(str(run_id), {})
        usage = {"input_tokens":None,"output_tokens":None,"total_tokens":None,
                 "cached_input_tokens":None,"reasoning_tokens":None,"model":record.get("model")}
        for group in getattr(response, "generations", []) or []:
            for generation in group:
                message = getattr(generation, "message", None)
                if message is None: continue
                metadata = getattr(message, "response_metadata", {}) or {}
                usage["model"] = metadata.get("model_name") or metadata.get("model") or usage["model"]
                meta = getattr(message, "usage_metadata", None) or {}
                if meta:
                    usage["input_tokens"] = meta.get("input_tokens")
                    usage["output_tokens"] = meta.get("output_tokens")
                    usage["total_tokens"] = meta.get("total_tokens")
                    in_details=meta.get("input_token_details",{}) or {}
                    out_details=meta.get("output_token_details",{}) or {}
                    usage["cached_input_tokens"]=in_details.get("cache_read",in_details.get("cached_tokens"))
                    usage["reasoning_tokens"]=out_details.get("reasoning")
                raw=metadata.get("token_usage",{}) or {}
                prompt_details=raw.get("prompt_tokens_details",{}) or {}
                completion_details=raw.get("completion_tokens_details",{}) or {}
                usage["input_tokens"] = usage["input_tokens"] if usage["input_tokens"] is not None else raw.get("prompt_tokens")
                usage["output_tokens"] = usage["output_tokens"] if usage["output_tokens"] is not None else raw.get("completion_tokens")
                usage["total_tokens"] = usage["total_tokens"] if usage["total_tokens"] is not None else raw.get("total_tokens")
                usage["cached_input_tokens"] = usage["cached_input_tokens"] if usage["cached_input_tokens"] is not None else prompt_details.get("cached_tokens")
                usage["reasoning_tokens"] = usage["reasoning_tokens"] if usage["reasoning_tokens"] is not None else completion_details.get("reasoning_tokens")
        usage.update({"query":record.get("query"),"duration_ms":round((time.perf_counter()-record.get("start",time.perf_counter()))*1000,1)})
        if usage["input_tokens"] is not None and usage["output_tokens"] is not None:
            cached=usage["cached_input_tokens"] or 0
            usage["cost_usd"]=round(((usage["input_tokens"]-cached)*INPUT_RATE+cached*CACHED_RATE+usage["output_tokens"]*OUTPUT_RATE)/1_000_000,10)
        else:
            usage["cost_usd"]=None
        self.calls.append(usage)

    def on_llm_error(self, error, *, run_id, **kwargs):
        record=self.started.pop(str(run_id),{})
        self.calls.append({"query":record.get("query"),"model":record.get("model"),"error_type":type(error).__name__,
                           "duration_ms":round((time.perf_counter()-record.get("start",time.perf_counter()))*1000,1),
                           "input_tokens":None,"output_tokens":None,"total_tokens":None,
                           "cached_input_tokens":None,"reasoning_tokens":None,"cost_usd":None})


def candidates(db_path, app_today):
    today=date.fromisoformat(app_today)
    con=sqlite3.connect(db_path); con.row_factory=sqlite3.Row
    try:
        rows=con.execute("""select o.customer_id,o.order_id,oi.item_id,p.name,s.delivered_at,p.returnable,
            p.warranty_class,oi.opened from orders o join order_items oi using(order_id)
            join products p using(product_id) left join shipments s using(order_id)""").fetchall()
        params={r["key"]:r["value"] for r in con.execute("select key,value from policy_parameters")}
        window=int(params["return_window_days"])
        speaker_rows=[r for r in rows if "speaker" in r["name"].casefold() and r["returnable"]!="no"]
        by_customer={}
        for row in speaker_rows: by_customer.setdefault(row["customer_id"],[]).append(row)
        unique=None; ambiguous=None
        for customer, items in by_customer.items():
            if len(items)>1 and ambiguous is None: ambiguous=(customer,items)
            if len(items)==1 and items[0]["delivered_at"] and 0 <= (today-date.fromisoformat(items[0]["delivered_at"][:10])).days <= window and unique is None:
                unique=(customer,items[0])
        return {"exact_customer":"C1001","unique_speaker":unique,"ambiguous_speaker":ambiguous}
    finally:
        con.close()


def main():
    settings=get_settings()
    if settings.llm_provider!="openai" or not settings.openai_api_key:
        print(json.dumps({"status":"BLOCKED","reason":"OpenAI is not configured"})); return 2
    budget=float(os.getenv("RESOLVEBOT_LIVE_TEST_BUDGET_USD","0.05"))
    original_db=settings.db_path; original_model=support_agent.ChatOpenAI
    capture=UsageCapture(); records=[]
    with tempfile.TemporaryDirectory(prefix="resolvebot-return-live-") as td:
        test_db=Path(td)/"shopsphere.db"; shutil.copy2(original_db,test_db); settings.db_path=test_db
        found=candidates(test_db,settings.app_today)
        support_agent.ChatOpenAI=lambda *args,**kwargs: original_model(*args,callbacks=[capture],**kwargs)
        from app.main import app
        from app.api.router import SESSIONS
        SESSIONS.clear()
        sock=socket.socket(); sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
        server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=port,log_level="critical",access_log=False))
        thread=threading.Thread(target=server.run,daemon=True); thread.start()
        until=time.monotonic()+15
        while not server.started and thread.is_alive() and time.monotonic()<until: time.sleep(.05)
        if not server.started:
            server.should_exit=True; thread.join(timeout=3); settings.db_path=original_db; support_agent.ChatOpenAI=original_model
            print(json.dumps({"status":"BLOCKED","reason":"Local API did not start"})); return 2

        def counts():
            con=sqlite3.connect(test_db)
            try: return {table:con.execute(f"select count(*) from {table}").fetchone()[0] for table in ("return_requests","support_tickets")}
            finally: con.close()

        def post(label,customer,sid,message):
            total_spend=sum(c.get("cost_usd") or 0 for c in capture.calls)
            if total_spend >= budget or len(capture.calls) >= 50:
                records.append({"query":label,"status":"BUDGET_LIMIT","route":None,"pending":None,"tools":[],
                    "db_before":counts(),"db_after":counts(),"duration_ms":0,"response_fields":False,
                    "budget_spend_usd":round(total_spend,10),"model_calls":[]})
                return None
            capture.active_query=label; start_count=len(capture.calls); before=counts()
            req=urllib.request.Request(f"http://127.0.0.1:{port}/chat",data=json.dumps({"session_id":sid,"message":message}).encode(),
                headers={"Content-Type":"application/json","X-Customer-Id":customer},method="POST")
            start=time.perf_counter(); status=None; payload={}
            try:
                with urllib.request.urlopen(req,timeout=35) as response: status=response.status; payload=json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                status=exc.code
                try: payload=json.loads(exc.read().decode())
                except Exception: payload={}
            except Exception as exc:
                status=None; payload={"transport_error":type(exc).__name__}
            elapsed=round((time.perf_counter()-start)*1000,1); after=counts(); tool_calls=payload.get("tool_calls",[])
            fields=all(key in payload for key in ("request_id","answer","route","citations","tool_calls","pending_confirmation"))
            record={"query":label,"status":status,"route":payload.get("route"),"pending":payload.get("pending_confirmation"),
                "tools":[{"name":t.get("name"),"status":t.get("status")} for t in tool_calls],
                "db_before":before,"db_after":after,"duration_ms":elapsed,"response_fields":fields,
                "answer_has_choices":bool(payload.get("answer") and ("order " in payload["answer"].lower() or "; " in payload["answer"])),
                "answer_requests_reason":bool(payload.get("answer") and ("changed your mind" in payload["answer"].lower() or "item is defective" in payload["answer"].lower())),
                "eligibility_tool_called":any(t.get("name")=="check_return_eligibility" for t in tool_calls),
                "write_tool_proposed":any(t.get("name")=="create_return_request" and t.get("status")=="pending_confirmation" for t in tool_calls),
                "model_calls":capture.calls[start_count:]}
            records.append(record); return record

        try:
            # The exact reported query: fresh session, then after real LLM-backed history.
            post("exact_query_fresh",found["exact_customer"],"fresh-exact-return","can i return my products?")
            post("history_seed",found["exact_customer"],"exact-after-history","What is the return window?")
            post("exact_query_after_history",found["exact_customer"],"exact-after-history","can i return my products?")

            # Full eligible flow on an account with one delivered speaker item.
            if found["unique_speaker"]:
                customer,item=found["unique_speaker"]; sid="return-flow-"+str(time.time_ns())
                post("flow_broad_question",customer,sid,"Can I return my products?")
                post("flow_item_choice",customer,sid,"My speaker.")
                post("flow_reason",customer,sid,"I changed my mind.")
                post("flow_action_request",customer,sid,"Please create the return.")
                post("flow_confirmation",customer,sid,"Yes, go ahead.")
                post("flow_duplicate_confirmation",customer,sid,"Yes, go ahead.")
                # Confirm that a different customer cannot reuse the same session.
                con=sqlite3.connect(test_db)
                try: other=con.execute("select customer_id from customers where customer_id<>? limit 1",(customer,)).fetchone()[0]
                finally: con.close()
                post("cross_customer_session",other,sid,"Yes, go ahead.")

                # An explicit eligibility question plus a defective reason must
                # remain read-only and must not create a proposal.
                defective_sid="defective-question-"+str(time.time_ns())
                post("defective_question",customer,defective_sid,"Can I return my speaker?")
                post("defective_reason",customer,defective_sid,"It is defective.")

            # A product name that belongs to multiple orders must remain ambiguous.
            if found["ambiguous_speaker"]:
                customer,items=found["ambiguous_speaker"]; sid="ambiguous-speaker-"+str(time.time_ns())
                post("ambiguous_broad_question",customer,sid,"Can I return my products?")
                post("ambiguous_speaker_choice",customer,sid,"My speaker.")
                if items:
                    chosen=items[0]
                    post("ambiguous_item_id_choice",customer,sid,chosen["item_id"])
                    post("ambiguous_defective_reason",customer,sid,"It is defective.")
        finally:
            server.should_exit=True; thread.join(timeout=10); settings.db_path=original_db
            support_agent.ChatOpenAI=original_model; SESSIONS.clear()

    original=sqlite3.connect(original_db)
    try: original_counts={t:original.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
    finally: original.close()
    total_input=sum(c.get("input_tokens") or 0 for c in capture.calls)
    total_output=sum(c.get("output_tokens") or 0 for c in capture.calls)
    total_tokens=sum(c.get("total_tokens") or 0 for c in capture.calls)
    result={"status":"COMPLETE","model":settings.openai_model,"budget_usd":budget,"records":records,"calls":capture.calls,
            "input_tokens":total_input,"output_tokens":total_output,"total_tokens":total_tokens,
            "call_count":len(capture.calls),"original_db_counts":original_counts}
    path=ROOT/"reports"/".return_live_capture.json"; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    expected_statuses=[r["status"]== (403 if r["query"]=="cross_customer_session" else 200) for r in records]
    print(json.dumps({"status":result["status"],"model":result["model"],"budget_usd":budget,"http_checks":len(records),
        "openai_calls":len(capture.calls),"input_tokens":total_input,"output_tokens":total_output,"total_tokens":total_tokens,
        "missing_usage":sum(c.get("input_tokens") is None or c.get("output_tokens") is None for c in capture.calls),
        "original_db_counts":original_counts,"expected_http_statuses":all(expected_statuses)},separators=(",",":")))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
