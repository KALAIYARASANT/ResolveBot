"""Verify SS-10195 return rejection through the live HTTP API on a DB copy."""
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
from datetime import date
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import uvicorn
from app.core.config import get_settings


def main():
    settings=get_settings(); original_db=settings.db_path
    original=sqlite3.connect(original_db); original.row_factory=sqlite3.Row
    try:
        row=original.execute('''select o.customer_id,o.order_id,o.status,oi.item_id,p.name,p.returnable,s.delivered_at
            from orders o join order_items oi using(order_id) join products p using(product_id)
            left join shipments s using(order_id) where o.customer_id='C1001' and o.order_id='SS-10195'
            and oi.item_id='SS-10195-1' ''').fetchone()
        if row is None: raise RuntimeError("Required C1001 test item was not present")
        target=dict(row)
        days=(date.fromisoformat(settings.app_today)-date.fromisoformat(target["delivered_at"][:10])).days
        window=int(original.execute("select value from policy_parameters where key='return_window_days'").fetchone()[0])
        original_counts={t:original.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
    finally: original.close()

    outcomes=[]
    with tempfile.TemporaryDirectory(prefix="resolvebot-expired-return-") as td:
        test_db=Path(td)/"shopsphere.db"; shutil.copy2(original_db,test_db); settings.db_path=test_db
        from app.main import app
        from app.api.router import SESSIONS
        SESSIONS.clear()
        sock=socket.socket(); sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
        server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=port,log_level="critical",access_log=False))
        thread=threading.Thread(target=server.run,daemon=True); thread.start()
        deadline=time.monotonic()+15
        while not server.started and thread.is_alive() and time.monotonic()<deadline: time.sleep(.05)
        if not server.started:
            server.should_exit=True; thread.join(timeout=3); settings.db_path=original_db
            print(json.dumps({"status":"BLOCKED","reason":"Local API did not start"})); return 2

        def counts():
            db=sqlite3.connect(test_db)
            try: return {t:db.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
            finally: db.close()

        def post(label,message):
            before=counts(); req=urllib.request.Request(f"http://127.0.0.1:{port}/chat",
                data=json.dumps({"session_id":"ss10195-"+label,"message":message}).encode(),
                headers={"Content-Type":"application/json","X-Customer-Id":"C1001"},method="POST")
            start=time.perf_counter(); status=None; payload={}
            try:
                with urllib.request.urlopen(req,timeout=35) as response: status=response.status; payload=json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                status=exc.code
                try: payload=json.loads(exc.read().decode())
                except Exception: payload={}
            outcomes.append({"case":label,"http_status":status,"answer":payload.get("answer"),
                "route":payload.get("route"),"pending_confirmation":payload.get("pending_confirmation"),
                "tool_calls":payload.get("tool_calls",[]),"db_before":before,"db_after":counts(),
                "http_duration_ms":round((time.perf_counter()-start)*1000,1)})

        try:
            order,item=target["order_id"],target["item_id"]
            post("eligibility_question",f"Can I return item {item} from order {order}? I changed my mind.")
            post("action_request",f"Please create a return for item {item} from order {order} because I changed my mind.")
        finally:
            server.should_exit=True; thread.join(timeout=10); settings.db_path=original_db; SESSIONS.clear()
    after=sqlite3.connect(original_db)
    try: original_after={t:after.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
    finally: after.close()
    print(json.dumps({"status":"COMPLETE","model":settings.openai_model,"app_today":settings.app_today,
        "return_window_days":window,"target":target,"days_since_delivery":days,"cases":outcomes,
        "original_db_before":original_counts,"original_db_after":original_after,
        "original_db_unchanged":original_counts==original_after},ensure_ascii=False,separators=(",",":")))
    return 0


if __name__=="__main__": raise SystemExit(main())
