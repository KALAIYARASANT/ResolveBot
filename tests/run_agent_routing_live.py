"""Targeted live OpenAI/LangGraph routing tests on a temporary SQLite copy."""
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

import uvicorn
from langchain_core.callbacks import BaseCallbackHandler

from app.core.config import get_settings
import app.agents.support_agent as support_agent


class UsageCapture(BaseCallbackHandler):
    def __init__(self):
        self.query = ""
        self.started = {}
        self.calls = []

    def _start(self, serialized, run_id):
        kwargs = (serialized or {}).get("kwargs", {})
        self.started[str(run_id)] = {"query": self.query, "start": time.perf_counter(),
                                    "model": kwargs.get("model") or kwargs.get("model_name")}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(serialized, run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(serialized, run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        row = self.started.pop(str(run_id), {})
        record = {"query": row.get("query"), "model": None, "duration_ms": None,
                  "input_tokens": None, "output_tokens": None, "total_tokens": None,
                  "cached_input_tokens": None, "reasoning_tokens": None, "generated_tool_names": []}
        for group in getattr(response, "generations", []) or []:
            for generation in group:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                metadata = getattr(message, "response_metadata", {}) or {}
                usage = getattr(message, "usage_metadata", {}) or {}
                raw = metadata.get("token_usage", {}) or {}
                input_details = usage.get("input_token_details", {}) or {}
                output_details = usage.get("output_token_details", {}) or {}
                prompt_details = raw.get("prompt_tokens_details", {}) or {}
                completion_details = raw.get("completion_tokens_details", {}) or {}
                record.update({
                    "model": metadata.get("model_name") or metadata.get("model") or row.get("model"),
                    "input_tokens": usage.get("input_tokens", raw.get("prompt_tokens")),
                    "output_tokens": usage.get("output_tokens", raw.get("completion_tokens")),
                    "total_tokens": usage.get("total_tokens", raw.get("total_tokens")),
                    "cached_input_tokens": input_details.get("cache_read", input_details.get("cached_tokens", prompt_details.get("cached_tokens"))),
                    "reasoning_tokens": output_details.get("reasoning", completion_details.get("reasoning_tokens")),
                    "generated_tool_names": [t.get("name") for t in (getattr(message, "tool_calls", []) or [])],
                })
        record["duration_ms"] = round((time.perf_counter() - row.get("start", time.perf_counter())) * 1000, 1)
        self.calls.append(record)

    def on_llm_error(self, error, *, run_id, **kwargs):
        row = self.started.pop(str(run_id), {})
        self.calls.append({"query": row.get("query"), "model": row.get("model"),
                           "duration_ms": round((time.perf_counter() - row.get("start", time.perf_counter())) * 1000, 1),
                           "error_type": type(error).__name__, "input_tokens": None, "output_tokens": None,
                           "total_tokens": None, "cached_input_tokens": None, "reasoning_tokens": None,
                           "generated_tool_names": []})


def choose_cases(db_path: Path, today_text: str):
    today = date.fromisoformat(today_text)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute("""SELECT o.customer_id,o.order_id,o.status,oi.item_id,oi.opened,
            p.name,p.returnable,p.warranty_class,s.delivered_at
            FROM orders o JOIN order_items oi USING(order_id) JOIN products p USING(product_id)
            LEFT JOIN shipments s USING(order_id) ORDER BY o.customer_id,o.order_id,oi.item_id""").fetchall()
        params = {r["key"]: r["value"] for r in con.execute("SELECT key,value FROM policy_parameters")}
        window = int(params["return_window_days"])
        eligible = None
        cancellation = None
        grouped = {}
        for row in rows:
            grouped.setdefault(row["customer_id"], {}).setdefault(row["order_id"], []).append(dict(row))
            if row["status"] == "processing" and cancellation is None:
                cancellation = dict(row)
            if row["status"] == "delivered" and row["returnable"] == "yes" and row["delivered_at"]:
                days = (today - date.fromisoformat(row["delivered_at"][:10])).days
                if 0 <= days <= window and eligible is None:
                    eligible = dict(row)
        complex_case = max(grouped.items(), key=lambda kv: (len(kv[1]), sum(map(len, kv[1].values()))))
        orders = list(complex_case[1].items())[:5]
        order_case = next((dict(item) for order in orders for item in order[1]), None)
        return {"eligible": eligible, "cancellation": cancellation, "complex_customer": complex_case[0],
                "complex_orders": [(oid, items) for oid, items in orders], "order": order_case}
    finally:
        con.close()


def main():
    settings = get_settings()
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        print(json.dumps({"status": "BLOCKED", "reason": "OpenAI is not configured"}))
        return 2
    budget = float(os.getenv("RESOLVEBOT_LIVE_TEST_BUDGET_USD", "0.05"))
    # Cost guard only; provider token counts below are never estimated.
    input_rate, output_rate = 0.10, 0.50
    original_db, original_model = settings.db_path, support_agent.ChatOpenAI
    capture, records = UsageCapture(), []
    with tempfile.TemporaryDirectory(prefix="resolvebot-routing-live-") as td:
        test_db = Path(td) / "shopsphere.db"
        shutil.copy2(original_db, test_db)
        settings.db_path = test_db
        cases = choose_cases(test_db, settings.app_today)
        support_agent.ChatOpenAI = lambda *args, **kwargs: original_model(*args, callbacks=[capture], **kwargs)
        from app.main import app
        from app.api.router import SESSIONS
        SESSIONS.clear()
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical", access_log=False))
        thread = threading.Thread(target=server.run, daemon=True); thread.start()
        deadline = time.monotonic() + 15
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            server.should_exit = True; thread.join(timeout=3)
            settings.db_path = original_db; support_agent.ChatOpenAI = original_model
            print(json.dumps({"status": "BLOCKED", "reason": "Local API did not start"}))
            return 2

        def counts():
            con = sqlite3.connect(test_db)
            try:
                return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                        for t in ("return_requests", "support_tickets")}
            finally:
                con.close()

        def post(label, customer, sid, message):
            spend = 0.0
            for call in capture.calls:
                if call.get("input_tokens") is not None and call.get("output_tokens") is not None:
                    spend += ((call["input_tokens"] - (call.get("cached_input_tokens") or 0)) * input_rate
                              + (call.get("cached_input_tokens") or 0) * 0.01
                              + call["output_tokens"] * output_rate) / 1_000_000
            if spend >= budget or len(capture.calls) >= 50:
                records.append({"query": label, "status": "BUDGET_LIMIT", "route": None, "tools": [], "model_calls": []})
                return None
            capture.query = label
            start_index = len(capture.calls)
            before = counts()
            request = urllib.request.Request(f"http://127.0.0.1:{port}/chat",
                data=json.dumps({"session_id": sid, "message": message}).encode(),
                headers={"Content-Type": "application/json", "X-Customer-Id": customer}, method="POST")
            start = time.perf_counter(); status = None; payload = {}
            try:
                with urllib.request.urlopen(request, timeout=35) as response:
                    status = response.status; payload = json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                status = exc.code
                try: payload = json.loads(exc.read().decode())
                except Exception: payload = {}
            except Exception as exc:
                payload = {"transport_error": type(exc).__name__}
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            record = {"query": label, "status": status, "route": payload.get("route"),
                      "pending_confirmation": payload.get("pending_confirmation"),
                      "answer_error": "OpenAIInvalidRequestError" in payload.get("answer", ""),
                      "tools": [{"name": t.get("name"), "status": t.get("status")} for t in payload.get("tool_calls", [])],
                      "generated_tool_call_count": sum(len(c.get("generated_tool_names", [])) for c in capture.calls[start_index:]),
                      "recorded_tool_call_count": len(payload.get("tool_calls", [])),
                      "response_fields_present": all(k in payload for k in ("request_id", "answer", "route", "citations", "tool_calls", "pending_confirmation")),
                      "http_duration_ms": elapsed, "db_before": before, "db_after": counts(),
                      "model_calls": capture.calls[start_index:]}
            records.append(record)
            return record

        try:
            order = cases["order"]
            post("broad_return_clarification", order["customer_id"], "broad-return-" + str(time.time_ns()),
                 "Can I return my products?")
            order_query = f"What is the status and delivery information for order {order['order_id']}?"
            post("order_status", order["customer_id"], "order-status-" + str(time.time_ns()), order_query)

            eligible = cases["eligible"]
            if eligible:
                text = f"Check if item {eligible['item_id']} in order {eligible['order_id']} is eligible for return because I changed my mind."
                post("return_eligibility", eligible["customer_id"], "return-check-" + str(time.time_ns()), text)
                post("warranty", eligible["customer_id"], "warranty-" + str(time.time_ns()),
                     f"Is the warranty active for order {eligible['order_id']}?")

            post("policy", order["customer_id"], "policy-" + str(time.time_ns()), "What is the current return window?")

            # Non-deterministic, multi-intent request with explicit IDs so it
            # traverses the real agent rather than the broad-return clarification.
            complex_lines = []
            for oid, items in cases["complex_orders"]:
                chosen = ", ".join(i["item_id"] for i in items)
                complex_lines.append(f"order {oid} items {chosen}")
            complex_query = ("For each of these orders/items: " + "; ".join(complex_lines)
                + ", report order status, warranty state and changed-mind return eligibility, and explain the current return window.")
            post("complex_multi_intent", cases["complex_customer"], "complex-" + str(time.time_ns()), complex_query)

            cancel = cases["cancellation"]
            if cancel:
                sid = "cancel-" + str(time.time_ns())
                proposal = post("cancellation_proposal", cancel["customer_id"], sid,
                                f"Please cancel order {cancel['order_id']}.")
                if proposal and proposal.get("pending_confirmation"):
                    con = sqlite3.connect(test_db)
                    try:
                        other = con.execute("SELECT customer_id FROM customers WHERE customer_id<>? LIMIT 1", (cancel["customer_id"],)).fetchone()[0]
                    finally:
                        con.close()
                    post("cross_customer_confirmation", other, sid, "Yes, go ahead.")
                    # Invalidate the cancellation after proposal but before yes.
                    con = sqlite3.connect(test_db)
                    try:
                        con.execute("UPDATE orders SET status='shipped' WHERE order_id=?", (cancel["order_id"],)); con.commit()
                    finally:
                        con.close()
                    post("confirmation_revalidation", cancel["customer_id"], sid, "Yes, go ahead.")
        finally:
            server.should_exit = True; thread.join(timeout=10)
            settings.db_path = original_db; support_agent.ChatOpenAI = original_model; SESSIONS.clear()

    original = sqlite3.connect(original_db)
    try:
        original_counts = {t: original.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                           for t in ("return_requests", "support_tickets")}
    finally:
        original.close()
    usage_calls = capture.calls
    result = {"status": "COMPLETE", "model": settings.openai_model, "budget_usd": budget,
              "records": records, "calls": usage_calls,
              "input_tokens": sum(c.get("input_tokens") or 0 for c in usage_calls),
              "output_tokens": sum(c.get("output_tokens") or 0 for c in usage_calls),
              "total_tokens": sum(c.get("total_tokens") or 0 for c in usage_calls),
              "original_db_counts": original_counts}
    path = ROOT / "reports" / ".agent_routing_live_capture.json"
    path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    status_ok = all(r.get("status") == (403 if r.get("query") == "cross_customer_confirmation" else 200) for r in records)
    no_invalid_request = all(not r.get("answer_error") and not any(c.get("error_type") == "OpenAIInvalidRequestError" for c in r.get("model_calls", [])) for r in records)
    tool_records_complete = all(r.get("generated_tool_call_count", 0) == r.get("recorded_tool_call_count", 0) for r in records if r.get("generated_tool_call_count", 0))
    print(json.dumps({"status": result["status"], "model": result["model"], "checks": len(records),
        "expected_http_statuses": status_ok, "no_invalid_request_error": no_invalid_request,
        "all_generated_calls_recorded": tool_records_complete,
        "openai_calls": len(usage_calls), "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"], "total_tokens": result["total_tokens"],
        "calls_missing_usage": sum(c.get("input_tokens") is None or c.get("output_tokens") is None for c in usage_calls),
        "original_db_counts": original_counts}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
