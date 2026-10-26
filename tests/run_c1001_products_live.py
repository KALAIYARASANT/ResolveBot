"""Replay the two C1001 product/order-count questions over live HTTP."""
from __future__ import annotations

import json
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn
from langchain_core.callbacks import BaseCallbackHandler

from app.core.config import get_settings
import app.agents.support_agent as support_agent


class Capture(BaseCallbackHandler):
    def __init__(self):
        self.label = ""
        self.active = {}
        self.calls = []

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        cfg = (serialized or {}).get("kwargs", {})
        self.active[str(run_id)] = {"query": self.label, "start": time.perf_counter(),
                                    "model": cfg.get("model") or cfg.get("model_name")}

    def on_llm_end(self, response, *, run_id, **kwargs):
        prior = self.active.pop(str(run_id), {})
        row = {"query": prior.get("query"), "model": prior.get("model"),
               "duration_ms": round((time.perf_counter()-prior.get("start", time.perf_counter()))*1000,1),
               "input_tokens": None, "output_tokens": None, "total_tokens": None,
               "cached_input_tokens": None, "reasoning_tokens": None, "generated_tool_names": []}
        for group in getattr(response, "generations", []) or []:
            for generation in group:
                msg = getattr(generation, "message", None)
                if msg is None: continue
                meta = getattr(msg, "usage_metadata", {}) or {}
                resp = getattr(msg, "response_metadata", {}) or {}
                raw = resp.get("token_usage", {}) or {}
                ind = meta.get("input_token_details", {}) or {}
                outd = meta.get("output_token_details", {}) or {}
                pd = raw.get("prompt_tokens_details", {}) or {}
                cd = raw.get("completion_tokens_details", {}) or {}
                row.update({"model":resp.get("model_name") or resp.get("model") or row["model"],
                    "input_tokens":meta.get("input_tokens",raw.get("prompt_tokens")),
                    "output_tokens":meta.get("output_tokens",raw.get("completion_tokens")),
                    "total_tokens":meta.get("total_tokens",raw.get("total_tokens")),
                    "cached_input_tokens":ind.get("cache_read",ind.get("cached_tokens",pd.get("cached_tokens"))),
                    "reasoning_tokens":outd.get("reasoning",cd.get("reasoning_tokens")),
                    "generated_tool_names":[t.get("name") for t in (getattr(msg,"tool_calls",[]) or [])]})
        self.calls.append(row)

    def on_llm_error(self, error, *, run_id, **kwargs):
        prior = self.active.pop(str(run_id), {})
        self.calls.append({"query":prior.get("query"),"model":prior.get("model"),
            "duration_ms":round((time.perf_counter()-prior.get("start",time.perf_counter()))*1000,1),
            "error_type":type(error).__name__,"input_tokens":None,"output_tokens":None,"total_tokens":None,
            "cached_input_tokens":None,"reasoning_tokens":None,"generated_tool_names":[]})


def main():
    settings=get_settings()
    if settings.llm_provider!="openai" or not settings.openai_api_key:
        print(json.dumps({"status":"BLOCKED","reason":"OpenAI is not configured"})); return 2
    original_db=settings.db_path; original_model=support_agent.ChatOpenAI
    con=sqlite3.connect(original_db); con.row_factory=sqlite3.Row
    try:
        db_summary={
            "orders_total":con.execute("select count(distinct order_id) from orders where customer_id='C1001'").fetchone()[0],
            "orders_by_status":{r["status"]:r["n"] for r in con.execute("select status,count(distinct order_id) n from orders where customer_id='C1001' group by status")},
            "item_rows":con.execute("select count(*) from orders o join order_items oi using(order_id) where o.customer_id='C1001'").fetchone()[0],
            "quantity_total":con.execute("select coalesce(sum(oi.qty),0) from orders o join order_items oi using(order_id) where o.customer_id='C1001'").fetchone()[0],
            "delivered_item_rows":con.execute("select count(*) from orders o join order_items oi using(order_id) where o.customer_id='C1001' and o.status='delivered'").fetchone()[0],
            "delivered_quantity":con.execute("select coalesce(sum(oi.qty),0) from orders o join order_items oi using(order_id) where o.customer_id='C1001' and o.status='delivered'").fetchone()[0],
            "cancelled_item_rows":con.execute("select count(*) from orders o join order_items oi using(order_id) where o.customer_id='C1001' and o.status='cancelled'").fetchone()[0],
            "cancelled_quantity":con.execute("select coalesce(sum(oi.qty),0) from orders o join order_items oi using(order_id) where o.customer_id='C1001' and o.status='cancelled'").fetchone()[0],
            "delivered_product_types":[r[0] for r in con.execute("select distinct p.name from orders o join order_items oi using(order_id) join products p using(product_id) where o.customer_id='C1001' and o.status='delivered' order by p.name")],
            "cancelled_orders":[r[0] for r in con.execute("select order_id from orders where customer_id='C1001' and status='cancelled'")],
        }
        original_counts={t:con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
    finally: con.close()

    capture=Capture(); results=[]
    with tempfile.TemporaryDirectory(prefix="resolvebot-c1001-live-") as td:
        test_db=Path(td)/"shopsphere.db"; shutil.copy2(original_db,test_db); settings.db_path=test_db
        support_agent.ChatOpenAI=lambda *args,**kwargs: original_model(*args,callbacks=[capture],**kwargs)
        from app.main import app
        from app.api.router import SESSIONS
        SESSIONS.clear()
        sock=socket.socket(); sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
        server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=port,log_level="critical",access_log=False))
        thread=threading.Thread(target=server.run,daemon=True); thread.start()
        deadline=time.monotonic()+15
        while not server.started and thread.is_alive() and time.monotonic()<deadline: time.sleep(.05)
        if not server.started:
            server.should_exit=True; thread.join(timeout=3); settings.db_path=original_db; support_agent.ChatOpenAI=original_model
            print(json.dumps({"status":"BLOCKED","reason":"Local API did not start"})); return 2

        def post(label,message,customer_id="C1001"):
            capture.label=label; start_count=len(capture.calls)
            req=urllib.request.Request(f"http://127.0.0.1:{port}/chat",
                data=json.dumps({"session_id":"c1001-product-count-audit","message":message}).encode(),
                headers={"Content-Type":"application/json","X-Customer-Id":customer_id},method="POST")
            began=time.perf_counter(); status=None; payload={}
            try:
                with urllib.request.urlopen(req,timeout=35) as response: status=response.status; payload=json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                status=exc.code
                try: payload=json.loads(exc.read().decode())
                except Exception: payload={}
            results.append({"query":label,"http_status":status,"route":payload.get("route"),
                "answer":payload.get("answer"),"tool_calls":payload.get("tool_calls",[]),
                "response_fields_present":all(k in payload for k in ("request_id","answer","route","citations","tool_calls","pending_confirmation")),
                "http_duration_ms":round((time.perf_counter()-began)*1000,1),
                "model_calls":capture.calls[start_count:]})

        try:
            post("products_question","what are products i purchased?")
            post("order_count_followup","check again how many orders? i think 10 only.")
            con=sqlite3.connect(test_db)
            try: other=con.execute("select customer_id from customers where customer_id<>'C1001' limit 1").fetchone()[0]
            finally: con.close()
            post("cross_customer_session","check again how many orders?",other)
        finally:
            server.should_exit=True; thread.join(timeout=10); settings.db_path=original_db
            support_agent.ChatOpenAI=original_model; SESSIONS.clear()

    db=sqlite3.connect(original_db)
    try: after_counts={t:db.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
    finally: db.close()
    for row in results:
        answer=(row.get("answer") or "").casefold()
        row["delivered_product_types_present"]=sum(name.casefold() in answer for name in db_summary["delivered_product_types"])
        row["distinct_product_types_total"]=len(db_summary["delivered_product_types"])
        row["mentions_cancelled_order"] = any(order.casefold() in answer for order in db_summary["cancelled_orders"])
        row["product_tool_recorded"] = any(call.get("name")=="get_orders" for call in row.get("tool_calls",[]))
    output={"status":"COMPLETE","model":settings.openai_model,"database":db_summary,
            "original_write_counts_before":original_counts,"original_write_counts_after":after_counts,
            "results":results,"model_calls":capture.calls,
            "totals":{"model_calls":len(capture.calls),
                "input_tokens":sum(c.get("input_tokens") or 0 for c in capture.calls),
                "output_tokens":sum(c.get("output_tokens") or 0 for c in capture.calls),
                "total_tokens":sum(c.get("total_tokens") or 0 for c in capture.calls),
                "missing_usage":sum(c.get("input_tokens") is None or c.get("output_tokens") is None for c in capture.calls)}}
    path=ROOT/"reports"/".c1001_products_live_capture.json"; path.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps({"status":output["status"],"model":output["model"],"database":db_summary,
        "queries":[{"query":r["query"],"http_status":r["http_status"],"route":r["route"],"tools":[c.get("name") for c in r["tool_calls"]],"calls":len(r["model_calls"]),"duration_ms":r["http_duration_ms"],"listed_delivered_types":r["delivered_product_types_present"]} for r in results],
        "totals":output["totals"],"original_writes_unchanged":original_counts==after_counts},separators=(",",":")))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
