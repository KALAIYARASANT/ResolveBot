"""Regression tests for the interview-readiness fixes.

No network and no real database: the chat model is replaced by a scripted stand-in and every write
goes to a temporary copy of the supplied SQLite file.
"""
from __future__ import annotations

import asyncio
import shutil
import sqlite3
import tempfile
import time
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.support_agent as agent
from app.api import router as router_module
from app.api.router import SESSIONS, Session
from app.core.config import ConfigurationError, Settings, get_settings
from app.core.ids import normalize_id, normalize_order_ids
from app.main import app
from app.services import business_rules
from app.services.policy_search import search_policy
from app.tools.support_tools import build_tools

ROOT = Path(__file__).resolve().parents[1]
NOT_FOUND = "I couldn't find that order on your account."


class ClosingConnection(sqlite3.Connection):
    """`with` commits and then closes (Windows cannot delete a temp DB that is still open)."""
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class ScriptedChat:
    """Stands in for ChatOpenAI: replays prepared AIMessages and records what it was given."""
    script: list = []
    bound: list = []
    kwargs: dict = {}
    seen: list = []

    def __init__(self, *args, **kwargs):
        ScriptedChat.kwargs = kwargs

    def bind_tools(self, tools):
        ScriptedChat.bound.append([t.name for t in tools])
        return self

    def invoke(self, messages):
        ScriptedChat.seen = messages
        return ScriptedChat.script.pop(0)


def tool_call(name, args, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class InterviewFixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="resolvebot-fixes-")
        self.db_path = Path(self.temp.name) / "shopsphere.db"
        shutil.copy2(ROOT / "data" / "shopsphere.db", self.db_path)
        with sqlite3.connect(self.db_path, factory=ClosingConnection) as db:  # the supplied tables start empty; this temp copy may not
            db.execute("DELETE FROM return_requests")
            db.execute("DELETE FROM support_tickets")
        self.settings = get_settings()
        self.saved = (self.settings.db_path, self.settings.app_today, business_rules.TODAY)
        self.settings.db_path = self.db_path
        self.settings.app_today = "2026-07-01"
        business_rules.TODAY = date(2026, 7, 1)
        ScriptedChat.script, ScriptedChat.bound, ScriptedChat.kwargs = [], [], {}
        SESSIONS.clear()

    def tearDown(self):
        self.settings.db_path, self.settings.app_today, business_rules.TODAY = self.saved
        SESSIONS.clear()
        self.temp.cleanup()

    # ---- helpers -------------------------------------------------------------------------------
    def db(self):
        db = sqlite3.connect(self.db_path, factory=ClosingConnection)
        db.row_factory = sqlite3.Row
        return db

    def count(self, table):
        with self.db() as db:
            return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def accounts(self):
        """(owner, one of their delivered orders, another customer, one of that customer's delivered orders)."""
        with self.db() as db:
            rows = db.execute("SELECT customer_id, order_id FROM orders WHERE status='delivered' ORDER BY order_id").fetchall()
        owner, own = rows[0]
        other, foreign = next((c, o) for c, o in rows if c != owner)
        return owner, own, other, foreign

    def returnable_item(self):
        with self.db() as db:
            row = db.execute("""SELECT o.customer_id, o.order_id, oi.item_id FROM orders o JOIN order_items oi USING(order_id)
                JOIN products p USING(product_id) JOIN shipments s USING(order_id)
                WHERE o.status='delivered' AND p.returnable='yes' AND s.delivered_at IS NOT NULL ORDER BY o.order_id LIMIT 1""").fetchone()
            db.execute("UPDATE shipments SET delivered_at=? WHERE order_id=?", ((date(2026, 7, 1) - timedelta(days=2)).isoformat(), row["order_id"]))
        return row["customer_id"], row["order_id"], row["item_id"]

    def chat(self, customer, message, history=None, pending=None, script=()):
        ScriptedChat.script = list(script)
        with patch.object(agent, "ChatOpenAI", ScriptedChat), patch.object(self.settings, "openai_api_key", "test-key"):
            return agent.run_chat(customer, history or [], message, pending)

    # ---- #3 configuration ------------------------------------------------------------------------
    def test_unsupported_provider_is_a_clear_configuration_error(self):
        settings = Settings()
        settings.llm_provider = "anthropic"
        with self.assertRaisesRegex(ConfigurationError, "LLM_PROVIDER='anthropic' is not supported"):
            settings.validate()
        settings.llm_provider, settings.openai_model = "openai", ""
        with self.assertRaisesRegex(ConfigurationError, "OPENAI_MODEL"):
            settings.validate()

    def test_model_comes_from_settings_and_reasoning_effort_is_opt_in(self):
        owner, *_ = self.accounts()
        with patch.object(self.settings, "openai_reasoning_effort", ""):
            self.chat(owner, "hello", script=[AIMessage(content="Hi!")])
        self.assertNotIn("reasoning_effort", ScriptedChat.kwargs)
        self.assertEqual(0, ScriptedChat.kwargs["temperature"])
        with patch.object(self.settings, "openai_reasoning_effort", "none"):
            self.chat(owner, "hello", script=[AIMessage(content="Hi!")])
        self.assertEqual("none", ScriptedChat.kwargs["reasoning_effort"])

    def test_the_model_is_told_the_business_date(self):
        owner, *_ = self.accounts()
        self.chat(owner, "What is the standard shipping fee?", script=[AIMessage(content="Hi")])
        self.assertIn("Today's date is 1 July 2026", ScriptedChat.seen[0].content)
        with patch.object(self.settings, "app_today", "2026-08-02"):
            self.chat(owner, "What is the standard shipping fee?", script=[AIMessage(content="Hi")])
        self.assertIn("Today's date is 2 August 2026", ScriptedChat.seen[0].content)

    # ---- #10 order id normalization --------------------------------------------------------------
    def test_order_ids_are_normalized_without_widening_access(self):
        self.assertEqual("Where is SS-10512 and SS-10458-1?", normalize_order_ids("Where is SS10512 and ss 10458-1?"))
        self.assertEqual("SS-10512", normalize_id(" ss10512 "))
        owner, own, other, foreign = self.accounts()
        tools = build_tools(owner, lambda _: None, allow_proposals=False)
        for spelling in (own.replace("-", ""), own.replace("-", " "), own.lower()):
            self.assertEqual(own, tools[0].invoke({"order_id": spelling})["orders"][0]["order_id"], spelling)
        for spelling in (foreign, foreign.replace("-", ""), foreign.replace("-", " ").lower()):
            self.assertEqual("ORDER_NOT_FOUND", tools[0].invoke({"order_id": spelling})["error_code"], spelling)
        item = f"{own}-1".replace("-", "", 1)
        self.assertNotEqual("ORDER_NOT_FOUND", business_rules.check_return_eligibility(owner, own.replace("-", ""), [item], "defective").get("error_code"))
        self.assertEqual("ORDER_NOT_FOUND", business_rules.check_return_eligibility(owner, foreign.replace("-", ""), [f"{foreign}-1"], "defective")["error_code"])

    def test_foreign_order_without_hyphen_gets_the_exact_reply(self):
        owner, own, other, foreign = self.accounts()
        reply = self.chat(owner, f"Where is order {foreign.replace('-', '')}?", script=[AIMessage(content="It is in transit to Mumbai.")])
        self.assertEqual((NOT_FOUND, "no_answer", [], False), (reply[0], reply[1], reply[3], reply[4]))

    # ---- #4 multi-question messages --------------------------------------------------------------
    def test_compound_messages_go_to_the_agent_loop_not_the_scripted_return_shortcut(self):
        owner, own, other, foreign = self.accounts()
        compound = f"Where is {foreign}, and can I return the speaker from {own}?"
        self.assertTrue(agent._is_compound(compound))
        self.assertTrue(agent._is_compound("Where is my order and can I return my charger?"))
        self.assertTrue(agent._is_compound("How long do refunds take? And what is the shipping fee?"))
        self.assertFalse(agent._is_compound(f"I want to return the speaker from {own}, I changed my mind."))
        with patch.object(agent, "_return_detail_turn", side_effect=AssertionError("shortcut must not run")):
            answer, route, calls, *_ = self.chat(
                owner, f"Where is {own}, and can I return the item from {own}-1?",
                script=[tool_call("get_orders", {"order_id": own}), AIMessage(content=f"Order {own} is delivered. What is the reason for the return?")])
        self.assertEqual([("get_orders", {"order_id": own})], [(c["name"], c["args"]) for c in calls])
        self.assertIn("reason", answer)

    def test_mixed_message_keeps_the_answerable_part_and_says_the_other_order_was_not_found(self):
        owner, own, other, foreign = self.accounts()
        answer, route, calls, citations, pending, _ = self.chat(
            owner, f"Where is {own} and where is {foreign}?",
            script=[tool_call("get_orders", {"order_id": own}), AIMessage(content=f"Order {own} is delivered.")])
        self.assertEqual("answer", route)
        self.assertIn(f"Order {own} is delivered.", answer)
        self.assertTrue(answer.endswith(NOT_FOUND))
        self.assertNotIn(foreign, answer)

    def test_an_order_id_the_model_invents_does_not_poison_a_valid_proposal(self):
        customer, order, item = self.returnable_item()
        answer, route, calls, _, pending, proposal = self.chat(
            customer, f"Please return {item} from {order}, I changed my mind.",
            script=[tool_call("create_return_request", {"order_id": "SS-00000", "item_ids": ["SS-00000-1"], "reason": "changed_mind"}, "a"),
                    tool_call("create_return_request", {"order_id": order, "item_ids": [item], "reason": "changed_mind"}, "b"),
                    AIMessage(content="I can create the return.")])
        self.assertTrue(pending)
        self.assertEqual(order, proposal["order_id"])
        self.assertNotIn(NOT_FOUND, answer)
        self.assertTrue(answer.endswith("Shall I go ahead?"))

    # ---- #5/#6/#8 retrieval and citations -------------------------------------------------------------
    def test_faq_only_questions_are_found_with_their_citation(self):
        for query, doc_id, section in (("Do you ship to Dubai?", "faq_general", "§6.1"),
                                       ("Can I change my delivery address?", "faq_general", "§5.1"),
                                       ("My phone stopped charging. Is it covered?", "warranty_policy", "§3.1"),
                                       ("warranty coverage for phone charging issue", "warranty_policy", "§3.1")):
            with self.subTest(query=query):
                result = search_policy(query)
                self.assertTrue(result["covered"])
                self.assertIn({"doc_id": doc_id, "section": section}, result["citations"])

    def test_unsupported_questions_are_uncovered_with_no_citations(self):
        for query in ("Do you price match competitors?", "Who won the cricket world cup?", "Write python code to reverse a list"):
            with self.subTest(query=query):
                self.assertEqual({"covered": False, "citations": [], "excerpts": []}, search_policy(query))

    def test_excerpts_flag_doc_type_and_rank_policy_before_faq(self):
        result = search_policy("What is ShopSphere Plus?")
        self.assertEqual("policy", result["excerpts"][0]["doc_type"])
        self.assertIn("faq", {x["doc_type"] for x in result["excerpts"]})  # still available to the model, flagged as FAQ

    def test_once_a_newer_document_is_in_force_it_replaces_the_older_section_it_changes(self):
        fee = lambda: {(c["doc_id"], c["section"]) for c in search_policy("What is the standard shipping fee?")["citations"]}
        with patch.object(self.settings, "app_today", "2026-07-31"):
            before = fee()
        with patch.object(self.settings, "app_today", "2026-08-01"):  # the update's effective date counts as in force
            after = fee()
            refunds = {(c["doc_id"], c["section"]) for c in search_policy("How long do refunds take?")["citations"]}
        self.assertIn(("shipping_policy", "§2.1"), before)
        self.assertNotIn("shipping_policy_update", {doc for doc, _ in before})
        self.assertIn(("shipping_policy_update", "§1.1"), after)
        self.assertNotIn(("shipping_policy", "§2.1"), after)
        self.assertIn(("refunds_policy", "§2.1"), refunds)  # unrelated sections of older documents are never dropped

    def test_sources_line_is_parsed_and_removed(self):
        text, named = agent._split_sources("No.\n\nSources: faq_general §6.1, shipping_policy §2.1 and §2.2")
        self.assertEqual("No.", text)
        self.assertEqual([("faq_general", "§6.1"), ("shipping_policy", "§2.1"), ("shipping_policy", "§2.2")], named)
        self.assertEqual(("Yes.", []), agent._split_sources("Yes.\nSources: none"))
        self.assertEqual(("Yes.", None), agent._split_sources("Yes."))

    def test_only_used_sections_are_cited(self):
        excerpts = [{"doc_id": "faq_general", "section": "§6.1", "text": "Do you ship outside India?: No. We deliver within India only."},
                    {"doc_id": "shipping_policy", "section": "§2.1", "text": "Standard shipping fee: Standard shipping costs 99 per order."}]
        candidates = [{"doc_id": x["doc_id"], "section": x["section"]} for x in excerpts]
        answer = "ShopSphere only delivers within India, so it does not ship outside India."
        # The model over-lists a fee section it did not use; the unrelated one is dropped.
        self.assertEqual(candidates[:1], agent._choose_citations([("faq_general", "§6.1"), ("shipping_policy", "§2.1")], candidates, excerpts, answer))
        # The model names nothing: the best-supported excerpt is cited.
        self.assertEqual(candidates[:1], agent._choose_citations([], candidates, excerpts, answer))
        # A search the answer did not rely on adds no citation, and invented sections are ignored.
        self.assertEqual([], agent._choose_citations(None, candidates, excerpts, "Order SS-10512 is in transit."))
        self.assertEqual([], agent._choose_citations([("returns_policy", "§9.9")], candidates, excerpts, "Order SS-10512 is in transit."))

    def test_answer_cites_only_what_it_used_end_to_end(self):
        owner, *_ = self.accounts()
        answer, route, calls, citations, *_ = self.chat(
            owner, "Do you ship to Dubai?",
            script=[tool_call("search_policy", {"query": "ship Dubai"}),
                    AIMessage(content="No. ShopSphere delivers within India only, so we do not ship outside India.\nSources: none")])
        self.assertEqual("answer", route)
        self.assertEqual([{"doc_id": "faq_general", "section": "§6.1"}], citations)
        self.assertNotIn("Sources", answer)
        self.assertEqual([("search_policy", "success")], [(c["name"], c["status"]) for c in calls])

    # ---- #7 routing ----------------------------------------------------------------------------
    def test_off_topic_requests_are_out_of_scope_with_nothing_attached(self):
        owner, *_ = self.accounts()
        for message in ("Write python code to reverse a list", "Who won the cricket world cup?"):
            with self.subTest(message=message):
                answer, route, calls, citations, pending, proposal = self.chat(owner, message, script=[AIMessage(content=agent._OUT_OF_SCOPE_REPLY)])
                self.assertEqual(("out_of_scope", [], [], False, None), (route, calls, citations, pending, proposal))
                self.assertEqual(agent._OUT_OF_SCOPE_REPLY, answer)

    def test_in_scope_but_unsupported_question_is_no_answer_without_citations(self):
        owner, *_ = self.accounts()
        answer, route, calls, citations, *_ = self.chat(
            owner, "Do you offer student discounts?",
            script=[tool_call("search_policy", {"query": "student discount"}), AIMessage(content="Probably yes.\nSources: shipping_policy §2.1")])
        self.assertEqual(("no_answer", []), (route, citations))
        self.assertEqual(agent._NOT_COVERED_REPLY, answer)
        self.assertIn("email", answer.lower())

    def test_a_refusal_is_retried_once_when_the_policy_files_do_cover_the_question(self):
        owner, *_ = self.accounts()
        answer, route, calls, citations, *_ = self.chat(
            owner, "Do you ship to Dubai?",
            script=[AIMessage(content=agent._OUT_OF_SCOPE_REPLY),
                    tool_call("search_policy", {"query": "ship Dubai"}),
                    AIMessage(content="No, we deliver within India only.\nSources: faq_general §6.1")])
        self.assertEqual("answer", route)
        self.assertEqual([{"doc_id": "faq_general", "section": "§6.1"}], citations)

    def test_write_tools_are_not_offered_on_read_only_turns(self):
        owner, *_ = self.accounts()
        self.chat(owner, "What is the warranty on laptops?", script=[AIMessage(content="Which item?")])
        self.assertEqual(["get_orders", "search_policy", "check_return_eligibility"], ScriptedChat.bound[-1])
        self.chat(owner, "Please open a warranty ticket", script=[AIMessage(content="Which order?")])
        self.assertEqual(5, len(ScriptedChat.bound[-1]))

    # ---- #6 warranty data the model needs -----------------------------------------------------------
    def test_get_orders_exposes_category_and_warranty_so_the_model_can_pick_the_item(self):
        owner, own, *_ = self.accounts()
        item = build_tools(owner, lambda _: None, allow_proposals=False)[0].invoke({"order_id": own})["orders"][0]["items"][0]
        for key in ("category", "warranty_end_date", "warranty_active", "item_id", "name"):
            self.assertIn(key, item)

    # ---- #9 conversation context ---------------------------------------------------------------------
    def test_it_refers_to_the_order_from_the_previous_turn(self):
        owner, own, *_ = self.accounts()
        history = [HumanMessage(content=f"Where is order {own}?"), AIMessage(content=f"Order {own} is delivered.")]
        reply = self.chat(owner, "When will it arrive?", history=history, script=[AIMessage(content=f"{own} was delivered already.")])
        self.assertEqual(("answer", f"{own} was delivered already."), (reply[1], reply[0]))
        asked = self.chat(owner, "When will it arrive?", history=[])
        self.assertEqual("clarify", asked[1])
        self.assertIn("Which one", asked[0])

    def test_a_policy_answer_about_returns_does_not_hijack_the_next_question(self):
        owner, *_ = self.accounts()
        statement = ("You can return most items within 30 days of the delivery date. Day 0 is the delivery date shown in "
                     "your order tracking, and Day 30 is the last day on which you can start a return.")
        history = [HumanMessage(content="How many days do I have to return something?"), AIMessage(content=statement)]
        for question in ("How long do refunds take?", "Do you ship to Dubai?"):
            self.assertIsNone(agent._return_detail_turn(owner, history, question, None), question)
        # A reply that really asked which item is still answered by the scripted clarification flow.
        asked = [HumanMessage(content="I want to return my charger"),
                 AIMessage(content="Which item would you like to return? Was the reason that you changed your mind, or that it is defective?")]
        self.assertIsNotNone(agent._return_detail_turn(owner, asked, "the charger, changed my mind", None))

    # ---- #11 reliability ---------------------------------------------------------------------------
    def test_duplicate_returns_and_cancellations_are_refused_without_changing_eligibility(self):
        customer, order, item = self.returnable_item()
        self.assertTrue(business_rules.check_return_eligibility(customer, order, [item], "changed_mind")["eligible"])
        self.assertTrue(business_rules.create_return_request(customer, order, [item], "changed_mind")["created"])
        # Eligibility is unchanged (pure business rules); the request step refuses a second request for the item.
        self.assertTrue(business_rules.check_return_eligibility(customer, order, [item], "changed_mind")["eligible"])
        again = business_rules.create_return_request(customer, order, [item], "changed_mind")
        self.assertEqual("ALREADY_REQUESTED", again["error_code"])
        self.assertEqual(1, self.count("return_requests"))
        proposal = build_tools(customer, lambda _: None)[3].invoke({"order_id": order, "item_ids": [item], "reason": "changed_mind"})
        self.assertEqual("ALREADY_REQUESTED", proposal["error_code"])
        with self.db() as db:
            customer2, order2 = db.execute("SELECT customer_id, order_id FROM orders WHERE status='processing' LIMIT 1").fetchone()
        self.assertTrue(business_rules.create_support_ticket(customer2, order2, "cancellation", "cancel")["created"])
        self.assertEqual("ALREADY_REQUESTED", business_rules.create_support_ticket(customer2, order2, "cancellation", "cancel again")["error_code"])
        self.assertEqual(1, self.count("support_tickets"))

    def test_rows_are_stamped_with_the_business_date_not_the_machine_date(self):
        customer, order, item = self.returnable_item()
        business_rules.create_return_request(customer, order, [item], "changed_mind")
        with self.db() as db:
            created = db.execute("SELECT created_at FROM return_requests").fetchone()[0]
        self.assertTrue(created.startswith("2026-07-01T"), created)

    def test_message_and_session_id_length_limits_are_http_400(self):
        owner, *_ = self.accounts()
        client = TestClient(app)
        headers = {"X-Customer-Id": owner}
        self.assertEqual(400, client.post("/chat", headers=headers, json={"session_id": "s", "message": "x" * 4001}).status_code)
        self.assertEqual(400, client.post("/chat", headers=headers, json={"session_id": "s" * 129, "message": "hi"}).status_code)
        with patch("app.api.router.run_chat", return_value=("ok", "answer", [], [], False, None)):
            self.assertEqual(200, client.post("/chat", headers=headers, json={"session_id": "s", "message": "x" * 4000}).status_code)

    def test_idle_sessions_expire_the_oldest_are_dropped_and_busy_ones_survive(self):
        now = time.monotonic()
        SESSIONS.update({"old": Session("C1", last_used=now - router_module.SESSION_TTL_SECONDS - 1),
                         "fresh": Session("C1", last_used=now)})
        router_module.evict_sessions(now)
        self.assertEqual({"fresh"}, set(SESSIONS))
        with patch.object(router_module, "MAX_SESSIONS", 2):
            SESSIONS.update({"a": Session("C1", last_used=now - 3), "b": Session("C1", last_used=now - 2)})
            router_module.evict_sessions(now)
            self.assertEqual({"fresh", "b"}, set(SESSIONS))  # least recently used ("a") went first
        busy = Session("C1", last_used=now - router_module.SESSION_TTL_SECONDS - 1)
        SESSIONS["busy"] = busy

        async def hold():
            await busy.lock.acquire()
            router_module.evict_sessions(now)
        asyncio.run(hold())
        self.assertIn("busy", SESSIONS)

    def test_stored_history_is_trimmed(self):
        owner, *_ = self.accounts()
        client = TestClient(app)
        with patch("app.api.router.run_chat", return_value=("ok", "answer", [], [], False, None)):
            for i in range(30):
                client.post("/chat", headers={"X-Customer-Id": owner}, json={"session_id": "long", "message": f"m{i}"})
        self.assertEqual(40, len(SESSIONS["long"].history))  # 30 exchanges = 60 messages, trimmed to the 40 most recent


if __name__ == "__main__":
    unittest.main(verbosity=2)
