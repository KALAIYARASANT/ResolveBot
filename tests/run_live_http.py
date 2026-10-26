"""Limited real OpenAI/LangGraph HTTP scenarios with provider usage capture."""
from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn
from langchain_core.callbacks import BaseCallbackHandler

from app.core.config import get_settings
from app.services import business_rules
import app.agents.support_agent as support_agent


INPUT_RATE = 0.10
CACHED_RATE = 0.01
OUTPUT_RATE = 0.50


def token_value(mapping, key):
    value = mapping.get(key) if isinstance(mapping, dict) else None
    return int(value) if isinstance(value, (int, float)) else None


def extract_usage(response):
    usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None,
             "cached_input_tokens": None, "reasoning_tokens": None, "model_name": None}
    for generation_list in getattr(response, "generations", []) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            meta = getattr(message, "response_metadata", {}) or {}
            usage["model_name"] = meta.get("model_name") or meta.get("model") or usage["model_name"]
            um = getattr(message, "usage_metadata", None) or {}
            if um:
                usage["input_tokens"] = token_value(um, "input_tokens")
                usage["output_tokens"] = token_value(um, "output_tokens")
                usage["total_tokens"] = token_value(um, "total_tokens")
                in_detail = um.get("input_token_details", {}) or {}
                out_detail = um.get("output_token_details", {}) or {}
                usage["cached_input_tokens"] = token_value(in_detail, "cache_read")
                if usage["cached_input_tokens"] is None:
                    usage["cached_input_tokens"] = token_value(in_detail, "cached_tokens")
                usage["reasoning_tokens"] = token_value(out_detail, "reasoning")
            if isinstance(meta.get("token_usage"), dict):
                raw = meta["token_usage"]
                usage["input_tokens"] = usage["input_tokens"] if usage["input_tokens"] is not None else token_value(raw, "prompt_tokens")
                usage["output_tokens"] = usage["output_tokens"] if usage["output_tokens"] is not None else token_value(raw, "completion_tokens")
                usage["total_tokens"] = usage["total_tokens"] if usage["total_tokens"] is not None else token_value(raw, "total_tokens")
                prompt_detail = raw.get("prompt_tokens_details", {}) or {}
                completion_detail = raw.get("completion_tokens_details", {}) or {}
                usage["cached_input_tokens"] = usage["cached_input_tokens"] if usage["cached_input_tokens"] is not None else token_value(prompt_detail, "cached_tokens")
                usage["reasoning_tokens"] = usage["reasoning_tokens"] if usage["reasoning_tokens"] is not None else token_value(completion_detail, "reasoning_tokens")
    if usage["model_name"] is None:
        usage["model_name"] = getattr(response, "llm_output", {}).get("model_name") if getattr(response, "llm_output", None) else None
    raw = getattr(response, "llm_output", {}).get("token_usage") if getattr(response, "llm_output", None) else None
    if isinstance(raw, dict):
        usage["input_tokens"] = usage["input_tokens"] if usage["input_tokens"] is not None else token_value(raw, "prompt_tokens")
        usage["output_tokens"] = usage["output_tokens"] if usage["output_tokens"] is not None else token_value(raw, "completion_tokens")
        usage["total_tokens"] = usage["total_tokens"] if usage["total_tokens"] is not None else token_value(raw, "total_tokens")
        prompt_detail = raw.get("prompt_tokens_details", {}) or {}
        completion_detail = raw.get("completion_tokens_details", {}) or {}
        usage["cached_input_tokens"] = usage["cached_input_tokens"] if usage["cached_input_tokens"] is not None else token_value(prompt_detail, "cached_tokens")
        usage["reasoning_tokens"] = usage["reasoning_tokens"] if usage["reasoning_tokens"] is not None else token_value(completion_detail, "reasoning_tokens")
    return usage


class UsageCapture(BaseCallbackHandler):
    def __init__(self):
        self.active_scenario = "unassigned"
        self.started = {}
        self.calls = []

    def _start(self, serialized, run_id):
        kwargs = (serialized or {}).get("kwargs", {})
        self.started[str(run_id)] = {"at": time.perf_counter(), "scenario": self.active_scenario,
                                    "model_name": kwargs.get("model") or kwargs.get("model_name")}

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(serialized, run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(serialized, run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        start = self.started.pop(str(run_id), {})
        usage = extract_usage(response)
        usage["model_name"] = usage["model_name"] or start.get("model_name")
        elapsed = max(0.0, time.perf_counter() - start.get("at", time.perf_counter()))
        usage.update({"scenario": start.get("scenario", self.active_scenario), "duration_ms": round(elapsed*1000, 1)})
        usage["cost_usd"] = None
        if usage["input_tokens"] is not None and usage["output_tokens"] is not None:
            cached = usage["cached_input_tokens"] or 0
            uncached = max(0, usage["input_tokens"] - cached)
            usage["cost_usd"] = round((uncached*INPUT_RATE + cached*CACHED_RATE + usage["output_tokens"]*OUTPUT_RATE)/1_000_000, 10)
        self.calls.append(usage)


def find_candidates(db_path, app_today):
    today = date.fromisoformat(app_today)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        window = int(con.execute("select value from policy_parameters where key='return_window_days'").fetchone()[0])
        return_rows = con.execute("""select o.customer_id,o.order_id,oi.item_id,s.delivered_at from orders o
            join order_items oi using(order_id) join products p using(product_id) join shipments s using(order_id)
            where o.status='delivered' and p.returnable='yes' and s.delivered_at is not null
            order by o.order_date desc""").fetchall()
        return_order = None; return_item_ids = []; seen_orders=set(); first_valid_return=None
        for candidate in return_rows:
            age=(today-date.fromisoformat(candidate["delivered_at"][:10])).days
            if not 0 <= age <= window or candidate["order_id"] in seen_orders: continue
            seen_orders.add(candidate["order_id"])
            ids=[r[0] for r in con.execute("""select oi.item_id from order_items oi join products p using(product_id)
                where oi.order_id=? and p.returnable='yes' order by oi.item_id""",(candidate["order_id"],))]
            if first_valid_return is None: first_valid_return=(candidate,ids)
            if len(ids)>=2:
                return_order=candidate; return_item_ids=ids; break
        if return_order is None and first_valid_return:
            return_order,return_item_ids=first_valid_return
        warranty_rows = con.execute("""select o.customer_id,o.order_id,s.delivered_at,p.warranty_class
            from orders o join order_items oi using(order_id) join products p using(product_id)
            join shipments s using(order_id) where o.status='delivered' and p.warranty_class!='none'
            and s.delivered_at is not null order by o.order_date desc""").fetchall()
        durations = {r["key"]: int(r["value"]) for r in con.execute("select key,value from policy_parameters")
                     if r["key"] in {"standard_warranty_months", "extended_warranty_months"}}
        from dateutil.relativedelta import relativedelta
        warranty = None
        for r in warranty_rows:
            months = durations["extended_warranty_months" if r["warranty_class"] == "extended" else "standard_warranty_months"]
            if today <= date.fromisoformat(r["delivered_at"][:10]) + relativedelta(months=months):
                warranty = r; break
        cancel = con.execute("select customer_id,order_id from orders where status='processing' order by order_date desc limit 1").fetchone()
        track = con.execute("""select o.customer_id,o.order_id from orders o join shipments s using(order_id)
            where s.tracking_no is not null and o.status in ('shipped','processing') order by o.order_date desc limit 1""").fetchone()
        foreign_order = con.execute("select order_id from orders where customer_id<>? limit 1", (track["customer_id"],)).fetchone() if track else None
        customers = [r[0] for r in con.execute("select customer_id from customers")]
        return dict(return_order=dict(return_order) if return_order else None,
                    return_item_ids=return_item_ids,
                    warranty=dict(warranty) if warranty else None,
                    cancel=dict(cancel) if cancel else None,
                    track=dict(track) if track else None,
                    customer=customers[0] if customers else None,
                    foreign_order_id=foreign_order["order_id"] if foreign_order else None,
                    other_customer=next((c for c in customers if track and c != track["customer_id"]), None))
    finally:
        con.close()


def main():
    settings = get_settings()
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        print(json.dumps({"live_status": "BLOCKED", "reason": "OpenAI provider/key is not configured"}))
        return 2
    budget = float(os.getenv("RESOLVEBOT_LIVE_TEST_BUDGET_USD", "0.05"))
    if budget <= 0:
        print(json.dumps({"live_status": "BLOCKED", "reason": "Live-test budget must be greater than zero"}))
        return 2
    original_db = settings.db_path
    original_model = support_agent.ChatOpenAI
    capture = UsageCapture()
    records = []
    request_start_counts = {}
    with tempfile.TemporaryDirectory(prefix="resolvebot-live-") as tempdir:
        temp_db = Path(tempdir)/"shopsphere.db"
        shutil.copy2(original_db, temp_db)
        settings.db_path = temp_db
        candidates = find_candidates(temp_db, settings.app_today)
        support_agent.ChatOpenAI = lambda *args, **kwargs: original_model(*args, callbacks=[capture], max_tokens=160, **kwargs)
        from app.main import app
        from app.api.router import SESSIONS
        SESSIONS.clear()

        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical", access_log=False))
        thread = threading.Thread(target=server.run, daemon=True); thread.start()
        deadline = time.monotonic()+20
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            server.should_exit = True; thread.join(timeout=3)
            settings.db_path = original_db; support_agent.ChatOpenAI = original_model
            print(json.dumps({"live_status":"BLOCKED","reason":"Local HTTP server did not start","budget_usd":budget}))
            return 2

        def db_counts():
            con=sqlite3.connect(temp_db)
            try: return {t:con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
            finally: con.close()

        def post(scenario, label, customer, session, message, expected):
            if sum((c.get("cost_usd") or 0) for c in capture.calls if c.get("cost_usd") is not None) >= budget:
                return False
            before = db_counts(); call_start = len(capture.calls); capture.active_scenario = label
            data = json.dumps({"session_id":session,"message":message}).encode()
            request = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=data,
                 headers={"Content-Type":"application/json","X-Customer-Id":customer}, method="POST")
            started = time.perf_counter(); status = None; payload = None
            try:
                with urllib.request.urlopen(request, timeout=35) as response:
                    status = response.status; payload = json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                status = exc.code
                try: payload=json.loads(exc.read().decode())
                except Exception: payload={}
            except Exception as exc:
                payload={"transport_error":type(exc).__name__}
            duration = round((time.perf_counter()-started)*1000,1); after=db_counts()
            calls = capture.calls[call_start:]
            tool_calls = payload.get("tool_calls",[]) if payload else []
            fields_ok = bool(payload and all(k in payload for k in ("request_id","answer","route","citations","tool_calls","pending_confirmation")))
            routes = {"answer","clarify","no_answer","out_of_scope"}
            expected_ok=expected(payload or {},before,after,tool_calls)
            checks={"http_200":status==200,"response_fields":fields_ok,
                    "route_valid":bool(payload and payload.get("route") in routes),
                    "expected":expected_ok}
            records.append({"scenario":scenario,"query":label,"http_status":status,"duration_ms":duration,
                "route":payload.get("route") if payload else None,"tool_calls":[{"name":t.get("name"),"status":t.get("status")} for t in tool_calls],
                "pending_confirmation":payload.get("pending_confirmation") if payload else None,
                "db_before":before,"db_after":after,"checks":checks,"expected_result":bool(expected_ok),
                "model_calls":calls,"transport_error":(payload or {}).get("transport_error")})
            return status == 200

        def use_tool(name):
            return lambda p,b,a,ts: any(t.get("name")==name and t.get("status") in {"success","pending_confirmation"} for t in ts)

        try:
            known_customer=candidates["customer"]
            for label, body, header, expected_status in [
                ("invalid_json",b"{",known_customer,400),
                ("missing_message",json.dumps({"session_id":"missing-message"}).encode(),known_customer,400),
                ("missing_customer_header",json.dumps({"session_id":"invalid-header","message":"Hello"}).encode(),None,401),
                ("unknown_customer_header",json.dumps({"session_id":"unknown-header","message":"Hello"}).encode(),"__unknown_customer__",401)]:
                req_headers={"Content-Type":"application/json"}
                if header is not None: req_headers["X-Customer-Id"]=header
                req=urllib.request.Request(f"http://127.0.0.1:{port}/chat",data=body,headers=req_headers,method="POST")
                try:
                    with urllib.request.urlopen(req,timeout=5) as response: probe_status=response.status
                except urllib.error.HTTPError as exc: probe_status=exc.code
                records.append({"scenario":"API validation","query":label,"http_status":probe_status,"duration_ms":None,
                    "route":None,"tool_calls":[],"pending_confirmation":None,"db_before":db_counts(),"db_after":db_counts(),
                    "checks":{"expected":probe_status==expected_status},"expected_result":probe_status==expected_status,"model_calls":[]})
            if candidates["track"]:
                track=candidates["track"]
                track_session="live-track-"+str(time.time_ns())
                post("order tracking","order_tracking",track["customer_id"],track_session,
                     f"Where is order {track['order_id']}?",use_tool("get_orders"))
                if candidates["other_customer"]:
                    other=candidates["other_customer"]
                    conflict=urllib.request.Request(f"http://127.0.0.1:{port}/chat",data=json.dumps({"session_id":track_session,"message":"Hello"}).encode(),headers={"Content-Type":"application/json","X-Customer-Id":other},method="POST")
                    try:
                        with urllib.request.urlopen(conflict,timeout=5) as response: conflict_status=response.status
                    except urllib.error.HTTPError as exc: conflict_status=exc.code
                    records.append({"scenario":"session isolation","query":"session_customer_reuse","http_status":conflict_status,"duration_ms":None,"route":None,"tool_calls":[],"pending_confirmation":None,"db_before":db_counts(),"db_after":db_counts(),"checks":{"expected":conflict_status==403},"expected_result":conflict_status==403,"model_calls":[]})
                if candidates["foreign_order_id"]:
                    post("foreign order isolation","foreign_order_lookup",track["customer_id"],"live-foreign-"+str(time.time_ns()),
                         f"Where is order {candidates['foreign_order_id']}?",
                         lambda p,b,a,ts: p.get("route")=="no_answer" and p.get("answer")=="I couldn't find that order on your account." and not p.get("citations") and any(t.get("name")=="get_orders" and t.get("status")=="error" for t in ts))
            post("RAG current shipping fee","current_shipping_fee",known_customer,"live-rag-"+str(time.time_ns()),
                 "What is the standard shipping fee?",lambda p,b,a,ts: any(x.get("name")=="search_policy" for x in ts) and any(c.get("doc_id")=="shipping_policy" for c in p.get("citations",[])))
            post("clarification","missing_order_id",known_customer,"live-clarify-"+str(time.time_ns()),
                 "Where is my order?",lambda p,b,a,ts: p.get("route")=="clarify" and any(t.get("name")=="get_orders" and t.get("status")=="success" for t in ts))
            if candidates["return_order"]:
                r=candidates["return_order"]; ids=candidates["return_item_ids"][:2] or [r["item_id"]]
                sid="live-return-"+str(time.time_ns()); msg=f"I want to return items {', '.join(ids)} from order {r['order_id']} because I changed my mind."
                proposed=post("return proposal","return_proposal",r["customer_id"],sid,msg,
                     lambda p,b,a,ts: bool(p.get("pending_confirmation")) and a["return_requests"]==b["return_requests"] and any(t.get("name")=="create_return_request" for t in ts))
                if proposed:
                    if len(ids)>1:
                        changed_item=ids[-1]
                        post("changed return details","return_details_changed",r["customer_id"],sid,
                             f"Actually, just item {changed_item}.",
                             lambda p,b,a,ts: bool(p.get("pending_confirmation")) and a["return_requests"]==b["return_requests"] and any(t.get("name")=="create_return_request" and t.get("args",{}).get("item_ids")==[changed_item] and t.get("status")=="pending_confirmation" for t in ts))
                    post("return confirmation","return_confirmation",r["customer_id"],sid,"Yes, go ahead.",
                         lambda p,b,a,ts: a["return_requests"]==b["return_requests"]+1)
                    post("duplicate confirmation","duplicate_return_confirmation",r["customer_id"],sid,"Yes, go ahead.",
                         lambda p,b,a,ts: a==b and not ts)
            if candidates["warranty"]:
                w=candidates["warranty"]; sid="live-warranty-"+str(time.time_ns())
                proposed=post("warranty ticket","warranty_proposal",w["customer_id"],sid,
                     f"Please open a warranty support ticket for order {w['order_id']} because the device stopped working.",
                     lambda p,b,a,ts: bool(p.get("pending_confirmation")) and a["support_tickets"]==b["support_tickets"] and any(t.get("name")=="create_support_ticket" for t in ts))
                if proposed:
                    post("warranty confirmation","warranty_confirmation",w["customer_id"],sid,"Confirm.",
                         lambda p,b,a,ts: a["support_tickets"]==b["support_tickets"]+1)
            if candidates["cancel"]:
                c=candidates["cancel"]; sid="live-cancel-"+str(time.time_ns())
                proposed=post("cancellation ticket","cancellation_proposal",c["customer_id"],sid,
                     f"I want to cancel order {c['order_id']} because I no longer need it.",
                     lambda p,b,a,ts: bool(p.get("pending_confirmation")) and a["support_tickets"]==b["support_tickets"] and any(t.get("name")=="create_support_ticket" for t in ts))
                if proposed:
                    post("cancellation confirmation","cancellation_confirmation",c["customer_id"],sid,"Please do.",
                         lambda p,b,a,ts: a["support_tickets"]==b["support_tickets"]+1)
            post("out-of-scope","unrelated_query",known_customer,"live-outscope-"+str(time.time_ns()),
                 "Can you write me a poem about mountains?",lambda p,b,a,ts: p.get("route")=="out_of_scope" and not ts and b==a)
        finally:
            server.should_exit=True; thread.join(timeout=10)
            settings.db_path=original_db; support_agent.ChatOpenAI=original_model
            SESSIONS.clear()

    result={"live_status":"COMPLETE","configured_model":settings.openai_model,"budget_usd":budget,
            "records":records,"call_count":len(capture.calls),"calls":capture.calls,
            "budget_exhausted":sum((c.get("cost_usd") or 0) for c in capture.calls if c.get("cost_usd") is not None)>=budget}
    raw_path=Path(os.getenv("RESOLVEBOT_LIVE_RESULT_PATH",str(ROOT/"reports"/".live_capture.json")))
    raw_path.parent.mkdir(parents=True,exist_ok=True)
    raw_path.write_text(json.dumps(result,indent=2,ensure_ascii=True),encoding="utf-8")
    total_cost=sum(c["cost_usd"] for c in capture.calls if c.get("cost_usd") is not None)
    total_input=sum(c["input_tokens"] for c in capture.calls if c.get("input_tokens") is not None)
    total_output=sum(c["output_tokens"] for c in capture.calls if c.get("output_tokens") is not None)
    total_usage=sum(c["total_tokens"] for c in capture.calls if c.get("total_tokens") is not None)
    print(json.dumps({"live_status":"COMPLETE","configured_model":settings.openai_model,"budget_usd":budget,
        "http_queries":len(records),"llm_calls":len(capture.calls),"input_tokens":total_input,
        "output_tokens":total_output,"total_tokens":total_usage,"estimated_cost_usd":round(total_cost,10),
        "requests_passed":sum(1 for r in records if r.get("expected_result")),
        "requests_failed":sum(1 for r in records if not r.get("expected_result")),
        "usage_rows_missing_provider_tokens":sum(1 for c in capture.calls if c.get("input_tokens") is None or c.get("output_tokens") is None),
        "budget_exhausted":result["budget_exhausted"]},separators=(",",":"),ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
