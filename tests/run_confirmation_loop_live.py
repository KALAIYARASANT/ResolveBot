"""Reproduce the reported return-confirmation loop through live /chat."""
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
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))

import uvicorn
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from app.core.config import get_settings
import app.agents.support_agent as support_agent


class UsageCapture(BaseCallbackHandler):
    def __init__(self): self.label=""; self.active={}; self.calls=[]
    def on_chat_model_start(self,serialized,messages,*,run_id,**kwargs):
        cfg=(serialized or {}).get("kwargs",{})
        self.active[str(run_id)]={"scenario":self.label,"start":time.perf_counter(),"model":cfg.get("model") or cfg.get("model_name")}
    def on_llm_end(self,response,*,run_id,**kwargs):
        prior=self.active.pop(str(run_id),{})
        row={"scenario":prior.get("scenario"),"model":prior.get("model"),"duration_ms":round((time.perf_counter()-prior.get("start",time.perf_counter()))*1000,1),
             "input_tokens":None,"output_tokens":None,"total_tokens":None,"cached_input_tokens":None,"reasoning_tokens":None}
        for group in getattr(response,"generations",[]) or []:
            for gen in group:
                msg=getattr(gen,"message",None)
                if msg is None: continue
                meta=getattr(msg,"usage_metadata",{}) or {}; rm=getattr(msg,"response_metadata",{}) or {}; raw=rm.get("token_usage",{}) or {}
                ind=meta.get("input_token_details",{}) or {}; outd=meta.get("output_token_details",{}) or {}
                pd=raw.get("prompt_tokens_details",{}) or {}; cd=raw.get("completion_tokens_details",{}) or {}
                row.update({"model":rm.get("model_name") or rm.get("model") or row["model"],
                    "input_tokens":meta.get("input_tokens",raw.get("prompt_tokens")),"output_tokens":meta.get("output_tokens",raw.get("completion_tokens")),
                    "total_tokens":meta.get("total_tokens",raw.get("total_tokens")),
                    "cached_input_tokens":ind.get("cache_read",ind.get("cached_tokens",pd.get("cached_tokens"))),
                    "reasoning_tokens":outd.get("reasoning",cd.get("reasoning_tokens"))})
        self.calls.append(row)
    def on_llm_error(self,error,*,run_id,**kwargs):
        prior=self.active.pop(str(run_id),{})
        self.calls.append({"scenario":prior.get("scenario"),"model":prior.get("model"),"duration_ms":round((time.perf_counter()-prior.get("start",time.perf_counter()))*1000,1),
            "error_type":type(error).__name__,"input_tokens":None,"output_tokens":None,"total_tokens":None,"cached_input_tokens":None,"reasoning_tokens":None})


def text_of(message):
    content=message.content
    return content if isinstance(content,str) else str(content)


def main():
    settings=get_settings()
    if settings.llm_provider!="openai" or not settings.openai_api_key:
        print(json.dumps({"status":"BLOCKED","reason":"OpenAI is not configured"})); return 2
    original_db=settings.db_path; original_model=support_agent.ChatOpenAI; capture=UsageCapture(); records=[]
    con=sqlite3.connect(original_db); con.row_factory=sqlite3.Row
    try:
        customer="C1001"
        item=con.execute("""select o.order_id,oi.item_id,p.name from orders o join order_items oi using(order_id)
            join products p using(product_id) where o.customer_id=? and p.name=? order by o.order_date desc limit 1""",
            (customer,"Voltix 30W Car Charger")).fetchone()
        if not item: raise RuntimeError("C1001 charger item not found")
        original_counts={t:con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
    finally: con.close()
    with tempfile.TemporaryDirectory(prefix="resolvebot-confirm-loop-") as td:
        test_db=Path(td)/"shopsphere.db"; shutil.copy2(original_db,test_db); settings.db_path=test_db
        support_agent.ChatOpenAI=lambda *args,**kwargs: original_model(*args,callbacks=[capture],**kwargs)
        from app.main import app
        from app.api.router import SESSIONS, Session
        SESSIONS.clear(); session_id="confirm-loop-"+str(uuid.uuid4())
        sock=socket.socket(); sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
        server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=port,log_level="critical",access_log=False))
        thread=threading.Thread(target=server.run,daemon=True); thread.start()
        deadline=time.monotonic()+15
        while not server.started and thread.is_alive() and time.monotonic()<deadline: time.sleep(.05)
        if not server.started:
            server.should_exit=True; thread.join(timeout=3); settings.db_path=original_db; support_agent.ChatOpenAI=original_model
            print(json.dumps({"status":"BLOCKED","reason":"Local API did not start"})); return 2

        def counts():
            db=sqlite3.connect(test_db)
            try: return {t:db.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
            finally: db.close()

        def post(label,message,session_key=session_id,customer_id=customer):
            capture.label=label; call_start=len(capture.calls); before=counts(); began=time.perf_counter()
            req=urllib.request.Request(f"http://127.0.0.1:{port}/chat",data=json.dumps({"session_id":session_key,"message":message}).encode(),
                headers={"Content-Type":"application/json","X-Customer-Id":customer_id},method="POST")
            status=None; payload={}
            try:
                with urllib.request.urlopen(req,timeout=35) as response: status=response.status; payload=json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                status=exc.code
                try: payload=json.loads(exc.read().decode())
                except Exception: payload={}
            session=SESSIONS.get(session_key)
            records.append({"step":label,"message":message,"http_status":status,"route":payload.get("route"),
                "answer":payload.get("answer"),"tool_calls":payload.get("tool_calls",[]),
                "pending_confirmation":payload.get("pending_confirmation"),"session_pending":session.pending if session else None,
                "confirmed_action_reply":session.confirmed_action_reply if session else None,
                "history":[{"role":type(m).__name__,"content":text_of(m)} for m in session.history] if session else [],
                "db_before":before,"db_after":counts(),"http_duration_ms":round((time.perf_counter()-began)*1000,1),
                "model_calls":capture.calls[call_start:]})

        try:
            post("eligibility_question_without_reason","Voltix 30W Car Charger return send back?")
            post("yes_1","yes")
            post("yes_2","yes")
            post("yes_return_please","yes return please")
            post("explicit_reason_after_clarification","I changed my mind")
            post("yes_accepts_general_offer","yes")
            post("no_cancels_pending_proposal","no")

            # Exercise the general-offer acceptance path using the exact style
            # of offer observed in the pre-fix live conversation.
            offer_session="general-offer-"+str(uuid.uuid4())
            SESSIONS[offer_session]=Session(customer_id=customer,history=[
                HumanMessage(content="Voltix 30W Car Charger return send back?"),
                AIMessage(content="The Voltix 30W Car Charger is eligible for return. Would you like me to start a return request? I'll ask you to confirm before submitting it."),
            ])
            post("yes_accepts_general_offer_but_requests_reason","yes",offer_session)
            post("offer_reason_explicit_changed_mind","I changed my mind",offer_session)
            post("yes_return_please_confirms_proposal","yes return please",offer_session)
            post("duplicate_yes_is_idempotent","yes return please",offer_session)

            # Defective reason, explicit action and the alternate clear
            # confirmation phrasing on a different eligible customer's item.
            defective_session="defective-"+str(uuid.uuid4())
            post("defective_action_proposal","Please create a return for item SS-10462-1 from order SS-10462 because it is defective.",
                 defective_session,"C1003")
            post("yes_go_ahead_confirms_defective_return","yes, go ahead",defective_session,"C1003")
            post("duplicate_yes_go_ahead_is_idempotent","yes, go ahead",defective_session,"C1003")

            # Expired and non-returnable items must stop before proposal creation.
            post("expired_return_action","Please return item SS-10195-1 from order SS-10195 because I changed my mind.",
                 "expired-"+str(uuid.uuid4()),"C1001")
            post("nonreturnable_action","Please return item SS-10476-1 from order SS-10476 because I changed my mind.",
                 "nonreturnable-"+str(uuid.uuid4()),"C1042")

            # A changed reason or item is a non-confirmation and must not
            # execute the previously proposed details.
            reason_change_session="reason-change-"+str(uuid.uuid4())
            post("reason_change_proposal","Please create a return for item SS-10427-1 from order SS-10427 because I changed my mind.",
                 reason_change_session,"C1004")
            post("reason_change_cancels_old_proposal","Actually, it is defective.",reason_change_session,"C1004")

            item_change_session="item-change-"+str(uuid.uuid4())
            post("item_change_proposal","Please create a return for item SS-10264-1 from order SS-10264 because I changed my mind.",
                 item_change_session,"C1010")
            post("item_change_cancels_old_proposal","Actually, use item SS-10264-2 instead.",item_change_session,"C1010")

            # Keep one ordinary policy request on the real agent path so the
            # post-fix HTTP run also exercises configured OpenAI/LangGraph.
            post("real_openai_policy_route","What is the current return window?",
                 "policy-"+str(uuid.uuid4()),"C1001")
        finally:
            server.should_exit=True; thread.join(timeout=10); settings.db_path=original_db
            support_agent.ChatOpenAI=original_model; SESSIONS.clear()
    db=sqlite3.connect(original_db)
    try: original_after={t:db.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
    finally: db.close()
    summary=[{k:r.get(k) for k in ("step","message","http_status","route","answer","tool_calls","pending_confirmation","session_pending","confirmed_action_reply","db_before","db_after","http_duration_ms","model_calls")} for r in records]
    print(json.dumps({"status":"COMPLETE","model":settings.openai_model,"session_id":session_id,
        "target_item":dict(item),"steps":summary,"model_calls_total":len(capture.calls),"provider_usage":capture.calls,
        "original_db_before":original_counts,"original_db_after":original_after,"original_db_unchanged":original_counts==original_after},ensure_ascii=False,separators=(",",":")))
    return 0


if __name__=="__main__": raise SystemExit(main())
