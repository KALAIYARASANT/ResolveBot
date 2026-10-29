"""Data-driven ResolveBot checks; all database writes use temporary copies."""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.config import get_settings
from app.core.dates import add_default_year
from app.services import business_rules
from app.services.policy_search import search_policy, _documents
from app.agents.support_agent import run_chat, _build_graph, _remove_inline_policy_citations
from app.tools.support_tools import build_tools
from app.main import app
from app.api.router import SESSIONS
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def rows(db: sqlite3.Connection, sql: str, args=()):
    return db.execute(sql, args).fetchall()


def count(db: sqlite3.Connection, table: str) -> int:
    return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


class ResolveBotScenarioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="resolvebot-test-")
        self.db_path = Path(self.temp.name) / "shopsphere.db"
        shutil.copy2(ROOT / "data" / "shopsphere.db", self.db_path)
        self.settings = get_settings()
        self.old_db_path = self.settings.db_path
        self.old_policies_path = self.settings.policies_path
        self.old_manifest_path = self.settings.manifest_path
        self.old_today = self.settings.app_today
        self.old_key = self.settings.openai_api_key
        self.settings.openai_api_key = ""
        self.old_rules_today = business_rules.TODAY
        self.settings.db_path = self.db_path
        self.settings.policies_path = ROOT / "data" / "policies.md"
        self.settings.manifest_path = ROOT / "data" / "manifest.json"
        self.settings.app_today = "2026-07-01"
        business_rules.TODAY = date.fromisoformat(self.settings.app_today)

    def tearDown(self):
        self.settings.db_path = self.old_db_path
        self.settings.policies_path = self.old_policies_path
        self.settings.manifest_path = self.old_manifest_path
        self.settings.app_today = self.old_today
        self.settings.openai_api_key = self.old_key
        business_rules.TODAY = self.old_rules_today
        self.temp.cleanup()

    def connect(self):
        db = sqlite3.connect(self.db_path, factory=ClosingConnection)
        db.row_factory = sqlite3.Row
        return db

    def table_count(self, table):
        with self.connect() as db:
            return count(db, table)

    def candidate(self, where="1=1", args=()):
        with self.connect() as db:
            return db.execute(f"""SELECT o.customer_id,o.order_id,o.status,oi.item_id,oi.opened,
                oi.defect_reported,p.returnable,p.warranty_class,s.delivered_at,s.eta,s.original_eta
                FROM orders o JOIN order_items oi USING(order_id) JOIN products p USING(product_id)
                LEFT JOIN shipments s USING(order_id) WHERE {where} ORDER BY o.order_id LIMIT 1""", args).fetchone()

    def test_return_window_boundary_and_dynamic_parameter(self):
        candidate = self.candidate("o.status='delivered' AND s.delivered_at IS NOT NULL AND p.returnable='yes'")
        self.assertIsNotNone(candidate)
        delivered = date.fromisoformat(candidate["delivered_at"][:10])
        with self.connect() as db:
            original = business_rules.policy_parameter(db, "return_window_days")
            db.execute("UPDATE policy_parameters SET value='30' WHERE key='return_window_days'")
            db.commit()
            business_rules.TODAY = delivered + timedelta(days=30)
            self.assertTrue(business_rules._check_return_eligibility(db, candidate["customer_id"], candidate["order_id"], [candidate["item_id"]], "changed_mind")["eligible"])
            business_rules.TODAY = delivered + timedelta(days=31)
            self.assertEqual("RETURN_WINDOW_EXPIRED", business_rules._check_return_eligibility(db, candidate["customer_id"], candidate["order_id"], [candidate["item_id"]], "changed_mind")["error_code"])
            db.execute("UPDATE policy_parameters SET value='35' WHERE key='return_window_days'")
            db.commit()
            business_rules.TODAY = delivered + timedelta(days=35)
            self.assertTrue(business_rules._check_return_eligibility(db, candidate["customer_id"], candidate["order_id"], [candidate["item_id"]], "changed_mind")["eligible"])
            business_rules.TODAY = delivered + timedelta(days=36)
            self.assertEqual("RETURN_WINDOW_EXPIRED", business_rules._check_return_eligibility(db, candidate["customer_id"], candidate["order_id"], [candidate["item_id"]], "changed_mind")["error_code"])
            db.execute("UPDATE policy_parameters SET value=? WHERE key='return_window_days'", (original,))

    def test_return_ownership_item_reason_and_no_invalid_write(self):
        candidate = self.candidate("o.status='delivered' AND s.delivered_at IS NOT NULL AND p.returnable='yes'")
        with self.connect() as db:
            business_today = date.fromisoformat(self.settings.app_today)
            db.execute("UPDATE shipments SET delivered_at=? WHERE order_id=?", ((business_today-timedelta(days=1)).isoformat(), candidate["order_id"]))
            db.commit()
        business_rules.TODAY = business_today
        before = self.table_count("return_requests")
        with self.connect() as db:
            missing = business_rules._check_return_eligibility(db, candidate["customer_id"], "missing-order", [candidate["item_id"]], "changed_mind")
            another_customer = db.execute("select customer_id from customers where customer_id<>? limit 1", (candidate["customer_id"],)).fetchone()[0]
            foreign = business_rules._check_return_eligibility(db, another_customer, candidate["order_id"], [candidate["item_id"]], "changed_mind")
            invalid_item = business_rules._check_return_eligibility(db, candidate["customer_id"], candidate["order_id"], ["not-an-item"], "changed_mind")
            foreign_item = db.execute("select item_id from order_items where order_id<>? limit 1", (candidate["order_id"],)).fetchone()[0]
            item_from_other_order = business_rules._check_return_eligibility(db, candidate["customer_id"], candidate["order_id"], [foreign_item], "changed_mind")
            invalid_reason = business_rules._check_return_eligibility(db, candidate["customer_id"], candidate["order_id"], [candidate["item_id"]], "other")
            duplicate = business_rules._check_return_eligibility(db, candidate["customer_id"], candidate["order_id"], [candidate["item_id"],candidate["item_id"]], "changed_mind")
        self.assertEqual("ORDER_NOT_FOUND", missing["error_code"])
        self.assertEqual("ORDER_NOT_FOUND", foreign["error_code"])
        self.assertEqual("INVALID_ITEM", invalid_item["error_code"])
        self.assertEqual("INVALID_ITEM", item_from_other_order["error_code"])
        self.assertEqual("INVALID_REASON", invalid_reason["error_code"])
        self.assertEqual("INVALID_ITEM", duplicate["error_code"])
        invalid_write = business_rules.create_return_request(candidate["customer_id"], candidate["order_id"], [candidate["item_id"]], "other")
        self.assertEqual("INVALID_REASON", invalid_write["error_code"])
        self.assertEqual(before, self.table_count("return_requests"))

    def test_returnable_classes_and_order_status(self):
        no = self.candidate("o.status='delivered' AND s.delivered_at IS NOT NULL AND p.returnable='no'")
        opened = self.candidate("o.status='delivered' AND s.delivered_at IS NOT NULL AND p.returnable='unopened_only' AND oi.opened=1")
        processing = self.candidate("o.status='processing'")
        shipped = self.candidate("o.status='shipped'")
        self.assertIsNotNone(no); self.assertIsNotNone(opened); self.assertIsNotNone(processing); self.assertIsNotNone(shipped)
        business_rules.TODAY = date.fromisoformat(opened["delivered_at"][:10]) + timedelta(days=1)
        with self.connect() as db:
            result_no = business_rules._check_return_eligibility(db, no["customer_id"], no["order_id"], [no["item_id"]], "defective")
            result_open = business_rules._check_return_eligibility(db, opened["customer_id"], opened["order_id"], [opened["item_id"]], "changed_mind")
            result_open_defective = business_rules._check_return_eligibility(db, opened["customer_id"], opened["order_id"], [opened["item_id"]], "defective")
            result_processing = business_rules._check_return_eligibility(db, processing["customer_id"], processing["order_id"], [processing["item_id"]], "defective")
            result_shipped = business_rules._check_return_eligibility(db, shipped["customer_id"], shipped["order_id"], [shipped["item_id"]], "defective")
        self.assertEqual("NON_RETURNABLE", result_no["error_code"])
        self.assertEqual("OPENED_NOT_RETURNABLE", result_open["error_code"])
        self.assertTrue(result_open_defective["eligible"])
        self.assertEqual("NOT_DELIVERED", result_processing["error_code"])
        self.assertEqual("NOT_DELIVERED", result_shipped["error_code"])

    def test_warranty_end_date_dynamic_months_and_calendar_arithmetic(self):
        standard = self.candidate("o.status='delivered' AND s.delivered_at IS NOT NULL AND p.warranty_class='standard'")
        self.assertIsNotNone(standard)
        with self.connect() as db:
            db.execute("UPDATE policy_parameters SET value='1' WHERE key='standard_warranty_months'")
            db.execute("UPDATE shipments SET delivered_at='2024-01-31' WHERE order_id=?", (standard["order_id"],))
            db.commit()
            end = date(2024, 1, 31) + business_rules.relativedelta(months=1)
            business_rules.TODAY = end
            on_end = business_rules._check_warranty(db, standard["customer_id"], standard["order_id"])
            business_rules.TODAY = end + timedelta(days=1)
            after = business_rules._check_warranty(db, standard["customer_id"], standard["order_id"])
            self.assertEqual("2024-02-29", on_end["items"][0]["warranty_end_date"])
            self.assertTrue(on_end["items"][0]["warranty_active"])
            self.assertEqual("WARRANTY_EXPIRED", after["error_code"])

    def test_extended_warranty_end_boundary_and_dynamic_duration(self):
        with self.connect() as db:
            original = business_rules.policy_parameter(db, "extended_warranty_months")
            db.execute("UPDATE policy_parameters SET value='1' WHERE key='extended_warranty_months'")
            db.commit()
            row = {"warranty_class":"extended", "delivered_at":"2024-01-31"}
            business_rules.TODAY = date(2024,2,28)
            before_end = business_rules.warranty_status(db, row)
            business_rules.TODAY = date(2024,2,29)
            on_end = business_rules.warranty_status(db, row)
            business_rules.TODAY = date(2024,3,1)
            after_end = business_rules.warranty_status(db, row)
            db.execute("UPDATE policy_parameters SET value='2' WHERE key='extended_warranty_months'")
            db.commit()
            business_rules.TODAY = date(2024,3,31)
            changed = business_rules.warranty_status(db, row)
            db.execute("UPDATE policy_parameters SET value=? WHERE key='extended_warranty_months'", (original,))
        self.assertTrue(before_end["warranty_active"])
        self.assertTrue(on_end["warranty_active"])
        self.assertFalse(after_end["warranty_active"])
        self.assertEqual("2024-03-31", changed["warranty_end_date"])
        self.assertTrue(changed["warranty_active"])

    def test_warranty_none_and_ticket_status_rules(self):
        no_warranty = self.candidate("o.status='delivered' AND s.delivered_at IS NOT NULL AND p.warranty_class='none'")
        processing = self.candidate("o.status='processing'")
        shipped = self.candidate("o.status='shipped'")
        self.assertIsNotNone(no_warranty); self.assertIsNotNone(processing); self.assertIsNotNone(shipped)
        with self.connect() as db:
            none = business_rules._check_warranty(db, no_warranty["customer_id"], no_warranty["order_id"])
            invalid = business_rules.validate_ticket(no_warranty["customer_id"], no_warranty["order_id"], "other", "summary")
            can_cancel = business_rules._validate_ticket(db, processing["customer_id"], processing["order_id"], "cancellation", "summary")
            cannot_cancel = business_rules._validate_ticket(db, shipped["customer_id"], shipped["order_id"], "cancellation", "summary")
            not_delivered = business_rules._check_warranty(db, processing["customer_id"], processing["order_id"])
        self.assertEqual("NO_WARRANTY", none["error_code"])
        self.assertEqual("INVALID_TICKET_TYPE", invalid["error_code"])
        self.assertTrue(can_cancel["eligible"])
        self.assertEqual("ORDER_ALREADY_SHIPPED", cannot_cancel["error_code"])
        self.assertEqual("NOT_DELIVERED", not_delivered["error_code"])

    def test_confirmation_is_next_message_same_session_and_idempotent(self):
        with self.connect() as db:
            customers=[r[0] for r in db.execute("select customer_id from customers order by customer_id limit 2")]
        customer, other = customers
        proposal={"kind":"return","order_id":"temporary","item_ids":["temporary-1"],"reason":"defective"}
        calls=[]
        def fake_run_chat(customer_id, history, message, pending):
            calls.append((customer_id,message,pending))
            if message == "prepare":
                return "Please confirm", "answer", [{"name":"create_return_request","args":{},"status":"pending_confirmation"}], [], True, proposal
            if pending and message.casefold() in {"yes","confirm","yes return please"}:
                return "Created", "answer", [{"name":"create_return_request","args":{},"status":"success"}], [], False, None
            return "No action", "answer", [], [], False, None
        SESSIONS.clear()
        try:
            with patch("app.api.router.run_chat", side_effect=fake_run_chat):
                client=TestClient(app)
                headers={"X-Customer-Id":customer}
                before=(self.table_count("return_requests"),self.table_count("support_tickets"))
                proposed=client.post("/chat",headers=headers,json={"session_id":"session-one","message":"prepare"})
                self.assertTrue(proposed.json()["pending_confirmation"])
                self.assertEqual(before,(self.table_count("return_requests"),self.table_count("support_tickets")))
                # Same proposal cannot be confirmed from another session.
                other_session=client.post("/chat",headers=headers,json={"session_id":"session-two","message":"yes"})
                self.assertFalse(other_session.json()["pending_confirmation"])
                self.assertIsNone(calls[-1][2])
                # A non-yes next message cancels the pending proposal.
                declined=client.post("/chat",headers=headers,json={"session_id":"session-one","message":"no"})
                self.assertFalse(declined.json()["pending_confirmation"])
                self.assertEqual(before,(self.table_count("return_requests"),self.table_count("support_tickets")))
                client.post("/chat",headers=headers,json={"session_id":"session-one","message":"prepare"})
                confirmed=client.post("/chat",headers=headers,json={"session_id":"session-one","message":"yes return please"})
                self.assertFalse(confirmed.json()["pending_confirmation"])
                self.assertEqual("success",confirmed.json()["tool_calls"][0]["status"])
                count_after_confirmation=len(calls)
                duplicate=client.post("/chat",headers=headers,json={"session_id":"session-one","message":"yes return please"})
                self.assertEqual("Created",duplicate.json()["answer"])
                self.assertEqual(count_after_confirmation,len(calls))
                rejected=client.post("/chat",headers={"X-Customer-Id":other},json={"session_id":"session-one","message":"yes"})
                self.assertEqual(403,rejected.status_code)
        finally:
            SESSIONS.clear()

    def test_no_unrelated_and_imperative_messages_never_execute_pending_action(self):
        with self.connect() as db:
            customer=db.execute("select customer_id from customers limit 1").fetchone()[0]
        proposal={"kind":"return","order_id":"temporary","item_ids":["temporary-1"],"reason":"defective"}
        before=(self.table_count("return_requests"),self.table_count("support_tickets"))
        with patch.object(self.settings,"llm_provider","disabled"), patch.object(self.settings,"openai_api_key",""):
            no_result=run_chat(customer,[],"no",proposal)
            unrelated=run_chat(customer,[],"What is the weather?",proposal)
            imperative=run_chat(customer,[],"just do it",proposal)
        for result in (no_result,unrelated,imperative):
            self.assertFalse(result[2])
            self.assertFalse(result[4])
            self.assertIsNone(result[5])
        self.assertEqual("clarify",imperative[1])
        self.assertIn("declined",no_result[0].facts["situation"])
        self.assertIn("not submitted",no_result[0].facts["situation"])
        self.assertEqual(before,(self.table_count("return_requests"),self.table_count("support_tickets")))

    def test_yearless_dates_use_2026_without_rewriting_explicit_year(self):
        self.assertIn("2026",add_default_year("Deliver it by July 3"))
        self.assertIn("2026",add_default_year("Deliver it by 3 July"))
        self.assertEqual("Deliver it by July 3, 2027",add_default_year("Deliver it by July 3, 2027"))

    def test_write_functions_use_correct_rows_and_revalidate(self):
        ret = self.candidate("o.status='delivered' AND s.delivered_at IS NOT NULL AND p.returnable='yes'")
        cancel = self.candidate("o.status='processing'")
        self.assertIsNotNone(ret); self.assertIsNotNone(cancel)
        today = date.fromisoformat(self.settings.app_today)
        business_rules.TODAY = today
        with self.connect() as db:
            db.execute("UPDATE shipments SET delivered_at=? WHERE order_id=?", ((today-timedelta(days=1)).isoformat(), ret["order_id"]))
            db.commit()
        with self.connect() as db:
            before_returns, before_tickets = count(db,"return_requests"), count(db,"support_tickets")
        ret_result = business_rules.create_return_request(ret["customer_id"], ret["order_id"], [ret["item_id"]], "changed_mind")
        ticket_result = business_rules.create_support_ticket(cancel["customer_id"], cancel["order_id"], "cancellation", "customer requested cancellation")
        self.assertTrue(ret_result.get("created")); self.assertTrue(ticket_result.get("created"))
        with self.connect() as db:
            self.assertEqual(before_returns+1, count(db,"return_requests")); self.assertEqual(before_tickets+1,count(db,"support_tickets"))
            row = db.execute("select order_id,customer_id,item_ids,reason,status,created_at from return_requests where request_id=?",(ret_result["request_id"],)).fetchone()
            self.assertEqual((ret["order_id"],ret["customer_id"],json.dumps([ret["item_id"]]),"changed_mind","requested"), tuple(row[:5]))
            siblings = db.execute("select count(*) from order_items where order_id=?",(ret["order_id"],)).fetchone()[0]
            self.assertGreaterEqual(siblings,1)
            self.assertTrue(row["created_at"])
            ticket = db.execute("select order_id,customer_id,ticket_type,summary,status,created_at from support_tickets where ticket_id=?",(ticket_result["ticket_id"],)).fetchone()
            self.assertEqual((cancel["order_id"],cancel["customer_id"],"cancellation","customer requested cancellation","open"), tuple(ticket[:5]))
            self.assertTrue(ticket["created_at"])
            db.execute("UPDATE policy_parameters SET value='0' WHERE key='return_window_days'")
            db.commit()
        failed = business_rules.create_return_request(ret["customer_id"], ret["order_id"], [ret["item_id"]], "changed_mind")
        self.assertEqual("RETURN_WINDOW_EXPIRED", failed["error_code"])
        with self.connect() as db:
            self.assertEqual(before_returns+1, count(db,"return_requests"))

    def test_policy_effective_dates_superseded_and_replacement_files(self):
        current_manifest = json.loads((ROOT/"data"/"manifest.json").read_text(encoding="utf-8"))
        tempdir = Path(self.temp.name)
        custom_policies = tempdir/"policies.md"
        custom_manifest = tempdir/"manifest.json"
        custom_policies.write_text((ROOT/"data"/"policies.md").read_text(encoding="utf-8") + "\n# Custom Policy\n\n**Document ID:** custom_doc · **Version:** 1 · **Effective date:** 2025-01-01 · **Status:** Current\n\n## §9 Custom topic\n\nReplacement marker phrase.\n",encoding="utf-8")
        current_manifest.append({"doc_id":"custom_doc","doc_type":"policy","version":"1","effective_date":"2025-01-01","status":"current"})
        custom_manifest.write_text(json.dumps(current_manifest),encoding="utf-8")
        self.settings.policies_path, self.settings.manifest_path = custom_policies, custom_manifest
        self.settings.app_today = "2026-07-31"
        before = search_policy("What is the standard shipping fee?")
        future = search_policy("What shipping fee changes from August 1 2026?")
        self.assertNotIn("shipping_policy_update", {c["doc_id"] for c in before["citations"]})
        self.assertIn("shipping_policy_update", {c["doc_id"] for c in future["citations"]})
        self.assertTrue(any("Replacement marker phrase" in x["text"] for x in search_policy("replacement marker phrase")["excerpts"]))
        self.settings.app_today = "2026-08-01"
        after = search_policy("What is the standard shipping fee?")
        # On the effective date the current update must be eligible and take priority.
        self.assertIn("shipping_policy_update", {c["doc_id"] for c in after["citations"]})
        changed = [dict(x) for x in current_manifest]
        for entry in changed:
            if entry["doc_id"] == "shipping_policy_update": entry["status"] = "superseded"
        custom_manifest.write_text(json.dumps(changed),encoding="utf-8")
        superseded = search_policy("What shipping fee changes from August 1 2026?")
        self.assertNotIn("shipping_policy_update", {c["doc_id"] for c in superseded["citations"]})

    def test_policy_topics_and_uncovered_question(self):
        supplied_policy=(ROOT/"data"/"policies.md").read_text(encoding="utf-8")
        supplied_manifest=json.loads((ROOT/"data"/"manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(10,len(list(_documents(supplied_policy))))
        self.assertEqual(10,len(supplied_manifest))
        cases = {
            "How do I return an item?":"returns_policy",
            "How long do refunds take?":"refunds_policy",
            "What warranty covers this device?":"warranty_policy",
            "What is the standard shipping fee?":"shipping_policy",
            "What happens when delivery is delayed?":"shipping_policy",
            "What if my item arrived damaged?":"damaged_or_defective_items",
            "Can I exchange an item?":"exchanges",
            "How can I cancel an order?":"order_cancellation",
            "What is ShopSphere Plus?":"shipping_policy",
            "What delivery tracking statuses are available?":"faq_general",
        }
        for query, doc_id in cases.items():
            with self.subTest(query=query):
                result=search_policy(query)
                self.assertTrue(result.get("covered"))
                self.assertIn(doc_id,{c["doc_id"] for c in result["citations"]})
        unsupported=search_policy("What is the color of the moon?")
        self.assertFalse(unsupported.get("covered"))
        self.assertEqual([],unsupported.get("citations"))
        self.assertEqual([],unsupported.get("excerpts"))
        current_policy=search_policy("What is the standard shipping fee?")
        self.assertNotIn("faq_general",{c["doc_id"] for c in current_policy["citations"]})

    def test_foreign_order_lookup_returns_same_not_found_result(self):
        with self.connect() as db:
            owner=db.execute("select customer_id,order_id from orders limit 1").fetchone()
            caller=db.execute("select customer_id from customers where customer_id<>? limit 1",(owner["customer_id"],)).fetchone()[0]
        tools=build_tools(caller,lambda _:None,allow_proposals=False)
        lookup=tools[0].invoke({"order_id":owner["order_id"]})
        missing=tools[0].invoke({"order_id":"SS-00000000"})
        self.assertEqual("ORDER_NOT_FOUND",lookup["error_code"])
        self.assertEqual(missing,lookup)

    def test_policy_citations_are_not_appended_to_customer_answer(self):
        answer="You can return most items within 30 days. (returns_policy §2.1, §3.1, §5.1)"
        self.assertEqual("You can return most items within 30 days.",_remove_inline_policy_citations(answer))

    def test_multiple_return_items_and_partial_item_selection(self):
        with self.connect() as db:
            order=db.execute("""select o.customer_id,o.order_id from orders o join order_items oi using(order_id)
                join products p using(product_id) join shipments s using(order_id)
                where o.status='delivered' and p.returnable='yes' and s.delivered_at is not null
                group by o.order_id having count(*)>=2 order by o.order_id limit 1""").fetchone()
            item_ids=[r[0] for r in db.execute("""select oi.item_id from order_items oi join products p using(product_id)
                where oi.order_id=? and p.returnable='yes' order by oi.item_id""",(order["order_id"],))]
            delivered=(date.fromisoformat(self.settings.app_today)-timedelta(days=1)).isoformat()
            db.execute("update shipments set delivered_at=? where order_id=?",(delivered,order["order_id"]))
            db.commit()
            business_rules.TODAY=date.fromisoformat(self.settings.app_today)
            many=business_rules._check_return_eligibility(db,order["customer_id"],order["order_id"],item_ids,"defective")
            one=business_rules._check_return_eligibility(db,order["customer_id"],order["order_id"],[item_ids[0]],"defective")
        self.assertTrue(many["eligible"])
        self.assertTrue(one["eligible"])
        self.assertGreaterEqual(len(item_ids),2)

    def test_missing_order_tracking_id_lists_matches_and_clarifies(self):
        with self.connect() as db:
            customer = db.execute("select customer_id from customers limit 1").fetchone()[0]
            order_ids = [r[0] for r in db.execute("select order_id from orders where customer_id=? order by order_date desc", (customer,))]
        answer, route, calls, citations, pending, proposal = run_chat(customer, [], "Where is my order?", None)
        self.assertEqual("clarify", route)
        self.assertEqual("get_orders", calls[0]["name"])
        self.assertEqual({}, calls[0]["args"])
        self.assertEqual(order_ids, answer.facts["orders_on_account"])
        self.assertEqual(["which order"], answer.facts["ask"])
        self.assertFalse(citations)
        self.assertFalse(pending)
        self.assertIsNone(proposal)

    def test_broad_return_query_lists_account_items_and_clarifies(self):
        with self.connect() as db:
            customer = db.execute("select customer_id from customers limit 1").fetchone()[0]
            product_names = {r[0] for r in db.execute("""select distinct p.name from orders o
                join order_items oi using(order_id) join products p using(product_id)
                where o.customer_id=?""", (customer,))}
        for history in ([], [HumanMessage(content="What is the return window?"), AIMessage(content="It depends on the item.")]):
            answer, route, calls, citations, pending, proposal = run_chat(
                customer, history, "can i return my products?", None
            )
            self.assertEqual("clarify", route)
            self.assertEqual("get_orders", calls[0]["name"])
            self.assertTrue(product_names)
            self.assertTrue(product_names)
            self.assertEqual(product_names, {item["name"] for item in answer.facts["account_items"]})
            self.assertEqual("which item they mean", answer.facts["ask"][0])
            self.assertIn("change their mind", answer.facts["ask"][1])  # the reason is asked together, never assumed
            self.assertFalse(citations)
            self.assertFalse(pending)
            self.assertIsNone(proposal)

    def test_product_return_question_does_not_invent_reason_or_proposal(self):
        with self.connect() as db:
            row=db.execute("""select o.customer_id from orders o join order_items oi using(order_id)
                join products p using(product_id) where o.customer_id='C1001' and p.name='Voltix 30W Car Charger' limit 1""").fetchone()
        customer=row[0]
        first=run_chat(customer,[],"Voltix 30W Car Charger return send back?",None)
        answer,route,calls,_citations,pending,proposal=first
        self.assertEqual("clarify",route)
        self.assertIn("change their mind",answer.facts["ask"][0])
        self.assertTrue(answer.facts["can_be_returned"])
        self.assertEqual(["get_orders","search_policy","check_return_eligibility","check_return_eligibility"],[call["name"] for call in calls])
        self.assertEqual(["changed_mind","defective"],[call["args"]["reason"] for call in calls[2:]])
        self.assertFalse(pending)
        self.assertIsNone(proposal)
        self.assertNotIn("reason",calls[0]["args"])

        history=[HumanMessage(content="Voltix 30W Car Charger return send back?"),
                 AIMessage(content=str(answer),additional_kwargs={"context":answer.context})]
        after_yes=run_chat(customer,history,"yes",None)
        self.assertEqual("clarify",after_yes[1])
        self.assertIn("not with a return reason",after_yes[0].facts["situation"])
        self.assertEqual(answer.context["selected"],after_yes[0].facts["item"])  # the item is kept, not re-searched
        self.assertEqual([],after_yes[2])
        self.assertFalse(after_yes[4])
        self.assertIsNone(after_yes[5])
        self.assertFalse(any(call["name"]=="create_return_request" for call in after_yes[2]))

    def test_product_name_answers_return_clarification_without_guessing_reason(self):
        customer="C1001"
        history=[HumanMessage(content="Can I return it?"),AIMessage(content=(
            "Are you asking whether a specific order is eligible, or would you like to start a return? "
            "For eligibility, please share the order ID and whether the reason is a change of mind or a defect."
        ))]
        identified=run_chat(customer,history,"car charger",None)
        self.assertEqual("clarify",identified[1])
        self.assertEqual("Voltix 30W Car Charger",identified[0].facts["item"]["name"])
        self.assertIn("change their mind",identified[0].facts["ask"][0])
        self.assertEqual(["get_orders","search_policy","check_return_eligibility","check_return_eligibility"],[call["name"] for call in identified[2]])
        self.assertFalse(identified[4])
        self.assertIsNone(identified[5])
        history.extend([HumanMessage(content="car charger"),
                        AIMessage(content=str(identified[0]),additional_kwargs={"context":identified[0].context})])
        yes_without_reason=run_chat(customer,history,"yes",None)
        self.assertEqual("clarify",yes_without_reason[1])
        self.assertIn("not with a return reason",yes_without_reason[0].facts["situation"])
        self.assertFalse(yes_without_reason[4])
        self.assertIsNone(yes_without_reason[5])
        self.assertFalse(any(call["name"]=="create_return_request" for call in yes_without_reason[2]))

    def test_yes_to_general_return_offer_becomes_proposal_then_compound_yes_confirms(self):
        with self.connect() as db:
            row=db.execute("""select o.customer_id from orders o join order_items oi using(order_id)
                join products p using(product_id) where o.customer_id='C1001' and p.name='Voltix 30W Car Charger' limit 1""").fetchone()
        customer=row[0]
        history=[
            HumanMessage(content="Can I return my Voltix 30W Car Charger? I changed my mind."),
            AIMessage(content="The Voltix 30W Car Charger is eligible for return. Would you like me to start a return request?"),
        ]
        before=self.table_count("return_requests")
        proposed=run_chat(customer,history,"yes",None)
        self.assertTrue(proposed[4])
        self.assertIsNotNone(proposed[5])
        self.assertEqual("changed_mind",proposed[5]["reason"])
        self.assertIn("pending_confirmation",[call["status"] for call in proposed[2]])
        self.assertEqual(before,self.table_count("return_requests"))
        history.extend([HumanMessage(content="yes"),AIMessage(content=proposed[0])])
        confirmed=run_chat(customer,history,"yes return please",proposed[5])
        self.assertFalse(confirmed[4])
        self.assertIsNone(confirmed[5])
        self.assertEqual("success",confirmed[2][0]["status"])
        self.assertEqual(before+1,self.table_count("return_requests"))

    def test_pending_return_reason_change_revalidates_and_reproposes(self):
        with self.connect() as db:
            row=db.execute("""select o.customer_id,o.order_id,oi.item_id from orders o join order_items oi using(order_id)
                where o.customer_id='C1004' and o.status='delivered' limit 1""").fetchone()
        proposal={"kind":"return","customer_id":row["customer_id"],"order_id":row["order_id"],
                  "item_ids":[row["item_id"]],"reason":"changed_mind"}
        before=self.table_count("return_requests")
        result=run_chat(row["customer_id"],[],"Actually, it is defective.",proposal)
        self.assertTrue(result[4])
        self.assertEqual("defective",result[5]["reason"])
        self.assertEqual(["check_return_eligibility","create_return_request"],[x["name"] for x in result[2]])
        self.assertEqual("pending_confirmation",result[2][1]["status"])
        self.assertEqual(before,self.table_count("return_requests"))

    def test_pending_return_item_change_revalidates_and_reproposes(self):
        with self.connect() as db:
            row=db.execute("""select o.customer_id,o.order_id,oi.item_id,oi2.item_id as other_item from orders o
                join order_items oi using(order_id) join order_items oi2 using(order_id)
                join shipments s using(order_id) where o.customer_id='C1010' and oi.item_id<>oi2.item_id
                and o.status='delivered' and julianday('2026-07-01')-julianday(s.delivered_at)<=30 limit 1""").fetchone()
        proposal={"kind":"return","customer_id":row["customer_id"],"order_id":row["order_id"],
                  "item_ids":[row["item_id"]],"reason":"changed_mind"}
        before=self.table_count("return_requests")
        result=run_chat(row["customer_id"],[],f"Actually, use item {row['other_item']} instead.",proposal)
        self.assertTrue(result[4])
        self.assertEqual([row["other_item"]],result[5]["item_ids"])
        self.assertEqual(["get_orders","check_return_eligibility","create_return_request"],[x["name"] for x in result[2]])
        self.assertEqual("pending_confirmation",result[2][-1]["status"])
        self.assertEqual(before,self.table_count("return_requests"))

    def test_every_generated_tool_call_gets_an_output_when_limit_is_reached(self):
        with self.connect() as db:
            customer = db.execute("select customer_id from customers limit 1").fetchone()[0]

        class FakeBoundModel:
            def invoke(self, messages):
                return AIMessage(content="", tool_calls=[
                    {"name":"get_orders","args":{},"id":f"fake-call-{index}","type":"tool_call"}
                    for index in range(9)
                ])

        class FakeModel:
            def bind_tools(self, tools):
                return FakeBoundModel()

            def invoke(self, messages):  # unbound: the model's closing question once the tool limit is reached
                return AIMessage(content="Which item do you mean?")

        graph, _ = _build_graph(customer, FakeModel(), {}, allow_proposals=False)
        result = graph.invoke({
            "messages":[HumanMessage(content="check these items")],
            "tool_calls":[],"citations":[],"call_count":2,
        }, config={"recursion_limit":18})
        outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual({f"fake-call-{i}" for i in range(9)}, {m.tool_call_id for m in outputs})
        self.assertEqual(9, len(result["tool_calls"]))
        self.assertEqual(4, sum(c["status"] == "success" for c in result["tool_calls"]))
        self.assertEqual(5, sum(c["status"] == "error" for c in result["tool_calls"]))
        self.assertEqual("Which item do you mean?", result["messages"][-1].content)

    def test_calls_emitted_after_cumulative_limit_are_recorded_and_not_reissued(self):
        with self.connect() as db:
            customer = db.execute("select customer_id from customers limit 1").fetchone()[0]

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def bind_tools(self, tools):
                return self.Bound(self)

            class Bound:
                def __init__(self, outer):
                    self.outer = outer

                def invoke(self, messages):
                    return self.outer.tool_batch()

            def invoke(self, messages):  # unbound: the model's closing question once the tool limit is reached
                return AIMessage(content="Which item do you mean?")

            def tool_batch(self):
                batch = 6 if self.calls == 0 else 2
                base = 0 if self.calls == 0 else 6
                self.calls += 1
                return AIMessage(content="", tool_calls=[
                    {"name":"get_orders","args":{},"id":f"limit-call-{i}","type":"tool_call"}
                    for i in range(base, base + batch)
                ])

        model = FakeModel()
        graph, _ = _build_graph(customer, model, {}, allow_proposals=False)
        result = graph.invoke({
            "messages":[HumanMessage(content="check several order details")],
            "tool_calls":[],"citations":[],"call_count":0,
        }, config={"recursion_limit":18})
        outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual({f"limit-call-{i}" for i in range(8)}, {m.tool_call_id for m in outputs})
        self.assertEqual(2, model.calls)
        self.assertEqual(8, len(result["tool_calls"]))
        self.assertEqual(6, sum(c["status"] == "success" for c in result["tool_calls"]))
        self.assertEqual(2, sum(c["status"] == "error" for c in result["tool_calls"]))
        self.assertEqual("Which item do you mean?", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
