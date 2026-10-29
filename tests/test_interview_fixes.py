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
            customer, f"Where is order {order}? Please return {item} from it, I changed my mind.",
            script=[tool_call("create_return_request", {"order_id": "SS-00000", "item_ids": ["SS-00000-1"], "reason": "changed_mind"}, "a"),
                    tool_call("create_return_request", {"order_id": order, "item_ids": [item], "reason": "changed_mind"}, "b"),
                    AIMessage(content="I can create the return.")])
        self.assertTrue(pending)
        self.assertEqual(order, proposal["order_id"])
        self.assertNotIn(NOT_FOUND, answer)
        self.assertTrue(answer.endswith("Shall I go ahead?"))

    def test_replies_are_written_by_the_model_and_required_text_is_guaranteed(self):
        customer, order, item = self.returnable_item()
        # The model writes the proposal but forgets the confirmation question: Python makes it the last sentence.
        answer, route, calls, citations, pending, proposal = self.chat(
            customer, f"Return {item} from {order}, I changed my mind.", script=[AIMessage(content="Yes, you can return it.")])
        self.assertTrue(answer.startswith("Yes, you can return it."))
        self.assertTrue(answer.endswith("Shall I go ahead?"))
        self.assertTrue(pending)
        sent = ScriptedChat.seen[-1].content  # the model was handed facts, not a drafted sentence
        self.assertIn('"eligible_for_return": true', sent)
        self.assertNotIn("_context", sent)
        history = [HumanMessage(content="Return it, I changed my mind."), AIMessage(content=str(answer))]
        # The model's confirmation omits the reference: Python adds it, and the saved row is the one named.
        confirmed, route, calls, *_ = self.chat(customer, "Yes, go ahead.", history=history, pending=proposal,
                                                script=[AIMessage(content="Your return request is saved.")])
        with self.db() as db:
            saved = db.execute("SELECT request_id FROM return_requests").fetchall()
        self.assertEqual(1, len(saved))
        self.assertIn(saved[0][0], confirmed)
        self.assertEqual([("create_return_request", "success")], [(c["name"], c["status"]) for c in calls])

    def test_a_model_that_rejects_temperature_zero_is_adapted_to_and_never_shows_raw_facts(self):
        customer, order, item = self.returnable_item()

        class ReasoningModel(ScriptedChat):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.reasoning = kwargs.get("reasoning_effort")

            def invoke(self, messages):
                if not self.reasoning:
                    raise ValueError("Unsupported value: 'temperature' does not support 0.0 with this model.")
                return super().invoke(messages)

        saved = self.settings.openai_reasoning_effort
        try:
            self.settings.openai_reasoning_effort = ""
            ScriptedChat.script = [AIMessage(content="Yes, you can return it.")]
            with patch.object(agent, "ChatOpenAI", ReasoningModel), patch.object(self.settings, "openai_api_key", "test-key"):
                answer = agent.run_chat(customer, [], f"Can I return {item} from {order}?", None)[0]
            self.assertEqual("Yes, you can return it.", answer)
            self.assertEqual("none", self.settings.openai_reasoning_effort)
            # A model that cannot be reached at all gets an apology, not the internal facts.
            class DownModel(ScriptedChat):
                def invoke(self, messages):
                    raise TimeoutError("down")
            with patch.object(agent, "ChatOpenAI", DownModel), patch.object(self.settings, "openai_api_key", "test-key"):
                down = agent.run_chat(customer, [], f"Can I return {item} from {order}?", None)[0]
            self.assertNotIn("situation", down)
            self.assertIn("Sorry", down)
        finally:
            self.settings.openai_reasoning_effort = saved

    def test_the_turn_carries_what_it_waits_for_so_a_bare_yes_is_not_a_new_request(self):
        customer, order, item = self.returnable_item()
        asked = self.chat(customer, f"Can I return {item} from {order}?", script=[AIMessage(content="Did you change your mind, or is it defective?")])
        self.assertEqual("clarify", asked[1])
        self.assertEqual("reason", asked[0].context["awaiting"])
        history = [HumanMessage(content="Can I return it?"),
                   AIMessage(content=str(asked[0]), additional_kwargs={"context": asked[0].context})]
        again = self.chat(customer, "yes", history=history, script=[AIMessage(content="Which reason applies?")])
        self.assertEqual(("clarify", []), (again[1], again[2]))  # no lookups, no list of the whole account
        self.assertEqual(asked[0].context["selected"], again[0].facts["item"])
        self.assertEqual(0, self.count("return_requests"))

    def test_a_bare_no_to_the_reason_question_is_verified_and_the_item_is_not_forgotten(self):
        customer, order, item = self.returnable_item()
        history = []

        def say(message, script=("Reply.",)):
            result = self.chat(customer, message, history=history, script=[AIMessage(content=s) for s in script])
            history.extend([HumanMessage(content=message),
                            AIMessage(content=str(result[0]), additional_kwargs={"context": getattr(result[0], "context", None)})])
            return result

        asked = say(f"Can I return {item} from {order}?")
        self.assertEqual("reason", asked[0].context["awaiting"])
        # "no" does not answer "changed your mind or defective?": ask whether to stop, keep the item, drop nothing.
        verify = say("no")
        self.assertEqual("clarify", verify[1])
        self.assertEqual("stop", verify[0].context["awaiting"])
        self.assertEqual(asked[0].context["selected"], verify[0].context["selected"])
        # "no" to "do you want to stop?" carries on and asks for the reason again.
        carry_on = say("no, i want return that")
        self.assertEqual("reason", carry_on[0].context["awaiting"])
        # A reason now resumes with the same item: no account-wide question, no new lookups of the whole account.
        resumed = say("yes, defective")
        self.assertEqual(asked[0].context["selected"], resumed[0].facts["item"])
        self.assertNotIn("account_items", resumed[0].facts)
        self.assertEqual(0, self.count("return_requests"))

    def test_confirming_the_stop_question_ends_the_return_without_a_write(self):
        customer, order, item = self.returnable_item()
        history = []
        for message in (f"Can I return {item} from {order}?", "no", "yes"):
            result = self.chat(customer, message, history=history, script=[AIMessage(content="Reply.")])
            history.extend([HumanMessage(content=message),
                            AIMessage(content=str(result[0]), additional_kwargs={"context": getattr(result[0], "context", None)})])
        self.assertIn("stop", result[0].facts["situation"])
        self.assertEqual({}, result[0].context)
        self.assertEqual(0, self.count("return_requests"))

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
                answer, route, calls, citations, pending, proposal = self.chat(owner, message, script=[AIMessage(content="Sorry, I can only help with ShopSphere orders and policies.\nRoute: out_of_scope")])
                self.assertEqual(("out_of_scope", [], [], False, None), (route, calls, citations, pending, proposal))
                self.assertEqual("Sorry, I can only help with ShopSphere orders and policies.", answer)  # the model's own words; the Route line is removed

    def test_in_scope_but_unsupported_question_is_no_answer_without_citations(self):
        owner, *_ = self.accounts()
        answer, route, calls, citations, *_ = self.chat(
            owner, "Do you offer student discounts?",
            script=[tool_call("search_policy", {"query": "student discount"}),
                    AIMessage(content="ShopSphere's policies do not cover student discounts. Please email the support team.\nSources: none")])
        self.assertEqual(("no_answer", []), (route, citations))
        self.assertIn("email", answer.lower())  # the model's words; Python only fixes the route and the empty citations

    def test_a_refusal_is_retried_once_when_the_policy_files_do_cover_the_question(self):
        owner, *_ = self.accounts()
        answer, route, calls, citations, *_ = self.chat(
            owner, "Do you ship to Dubai?",
            script=[AIMessage(content="That is outside what I can help with.\nRoute: out_of_scope"),
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
        self.assertEqual(["which order"], asked[0].facts["ask"])

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

    # ---- conversation state and intent routing ---------------------------------------------------------
    def session(self, customer):
        """A chat in one session: each call sends the message with the history so far, as the router does."""
        history = []

        def say(message, script=()):
            result = self.chat(customer, message, history=history, script=list(script))
            history.extend([HumanMessage(content=message),
                            AIMessage(content=str(result[0]), additional_kwargs={"context": getattr(result[0], "context", None)})])
            return result
        return say

    def test_unrelated_policy_questions_do_not_get_the_pending_return_clarification(self):
        customer, order, item = self.returnable_item()
        say = self.session(customer)
        asked = say(f"Can I return {item} from {order}?", [AIMessage(content="Did you change your mind, or is it defective?")])
        self.assertEqual("reason", asked[0].context["awaiting"])
        fee_query = "free shipping"

        free = say("Do you offer free shipping today?", [tool_call("search_policy", {"query": fee_query}),
                                                         AIMessage(content="Yes, on orders of ₹2,999 or more, and always for Plus members.\nSources: shipping_policy §2.1, shipping_policy §2.2")])
        self.assertEqual("answer", free[1])
        self.assertEqual([("search_policy", {"query": fee_query})], [(c["name"], c["args"]) for c in free[2]])
        self.assertEqual([{"doc_id": "shipping_policy", "section": "§2.1"}, {"doc_id": "shipping_policy", "section": "§2.2"}], free[3])
        self.assertIsNone(getattr(free[0], "context", None))  # nothing is left waiting for a return reason
        self.assertTrue(any("previous question to the customer is closed" in str(m.content) for m in ScriptedChat.seen if m.type == "system"))

        fee = say("What is the current standard shipping fee?", [tool_call("search_policy", {"query": "standard shipping fee"}),
                                                                 AIMessage(content="Standard shipping costs ₹99 per order.\nSources: shipping_policy §2.1")])
        self.assertEqual("answer", fee[1])
        self.assertEqual([{"doc_id": "shipping_policy", "section": "§2.1"}], fee[3])  # the current fee, not the 1 August 2026 notice
        self.assertIn("₹99", fee[0])
        self.assertEqual(0, self.count("return_requests"))

    def test_shipping_policy_is_retrieved_by_app_today_even_when_a_future_notice_matches_better(self):
        for query in ("current standard shipping fee", "standard shipping fee", "free shipping today",
                      "standard shipping fee from 1 August 2026 and current fee"):
            with self.subTest(query=query):
                result = search_policy(query)
                current = [x for x in result["excerpts"] if x["doc_id"] == "shipping_policy" and x["section"] == "§2.1"]
                self.assertEqual(1, len(current), result["citations"])
                self.assertIn("₹99", current[0]["text"])
                self.assertTrue(all(x["future"] or x["effective_date"] <= "2026-07-01" for x in result["excerpts"]))
        # A plain present-tense question never retrieves the notice that is not yet in force.
        self.assertFalse(any(x["future"] for x in search_policy("current standard shipping fee")["excerpts"]))

    def test_a_reason_reply_still_continues_the_return_flow(self):
        customer, order, item = self.returnable_item()
        say = self.session(customer)
        asked = say(f"Can I return {item} from {order}?", [AIMessage(content="Did you change your mind, or is it defective?")])
        self.assertEqual("reason", asked[0].context["awaiting"])
        defective = say("defective", [AIMessage(content="It can be returned for that reason. Shall I submit it?")])
        self.assertEqual(asked[0].context["selected"], defective[0].facts["item"])
        self.assertEqual(("defective", True), (defective[0].facts["reason"], defective[0].facts["eligible_for_return"]))
        self.assertIn("check_return_eligibility", [c["name"] for c in defective[2]])
        self.assertEqual(0, self.count("return_requests"))  # a reason alone never writes

    def order_dates(self):
        with self.db() as db:
            return db.execute("SELECT customer_id, substr(order_date,1,10) AS day, count(*) AS n FROM orders GROUP BY 1, 2").fetchall()

    def test_order_date_question_searches_only_the_customers_own_orders(self):
        rows = self.order_dates()
        owner, day, _n = next(r for r in rows if r["n"] == 1)
        elsewhere = {r["customer_id"] for r in rows if r["day"] == day}
        result = self.chat(owner, f"I ordered on {day}. What product was that?", script=[AIMessage(content="That was your order.")])
        facts = result[0].facts
        self.assertEqual(("answer", [("get_orders", {"order_date": day})]), (result[1], [(c["name"], c["args"]) for c in result[2]]))
        with self.db() as db:
            expected = db.execute("SELECT order_id FROM orders WHERE customer_id=? AND substr(order_date,1,10)=?", (owner, day)).fetchone()[0]
        self.assertEqual(expected, facts["order"]["order_id"])
        self.assertTrue(facts["order"]["products"])
        # Another customer asking for a date that only someone else ordered on gets "not found", never that order.
        other = next(r["customer_id"] for r in rows if r["customer_id"] != owner and r["customer_id"] not in elsewhere)
        foreign = self.chat(other, f"I ordered on {day}. What product was that?", script=[AIMessage(content="unused")])
        self.assertEqual((NOT_FOUND, "no_answer"), (str(foreign[0]), foreign[1]))
        self.assertNotIn(expected, str(foreign[0]))

    def test_order_date_with_several_matching_orders_asks_which_one(self):
        rows = self.order_dates()
        owner, day, count = next(r for r in rows if r["n"] > 1)
        result = self.chat(owner, f"I ordered on {day}. What product was that?", script=[AIMessage(content="Which order do you mean?")])
        self.assertEqual("clarify", result[1])
        self.assertEqual(["which order they mean"], result[0].facts["ask"])
        self.assertEqual(count, len(result[0].facts["matching_orders"]))
        with self.db() as db:
            mine = {r[0] for r in db.execute("SELECT order_id FROM orders WHERE customer_id=?", (owner,))}
        self.assertTrue({o["order_id"] for o in result[0].facts["matching_orders"]} <= mine)

    # ---- bug 1: the shipping policy that is in force on APP_TODAY ----------------------------------------
    def excerpts_at(self, today, query):
        with patch.object(self.settings, "app_today", today):
            return search_policy(query)["excerpts"]

    def test_current_shipping_policy_on_app_today_for_equivalent_questions(self):
        for query in ("What is the current shipping fee?", "Do you offer free shipping today?", "What is the free-shipping threshold?",
                      "shipping fee 2026-07-01", "shipping fee free shipping threshold change delivery address"):
            with self.subTest(query=query):
                excerpts = self.excerpts_at("2026-07-01", query)
                fee = next(x for x in excerpts if (x["doc_id"], x["section"]) == ("shipping_policy", "§2.1"))
                self.assertIn("₹99", fee["text"])
                self.assertIn("₹2,999", fee["text"])
                self.assertFalse(any(x["future"] or x["doc_id"] == "shipping_policy_update" for x in excerpts))
                if "free" in query:
                    self.assertIn(("shipping_policy", "§2.2"), [(x["doc_id"], x["section"]) for x in excerpts])  # Plus members
        address = self.excerpts_at("2026-07-01", "shipping fee free shipping threshold change delivery address")
        self.assertIn(("faq_general", "§5.1"), [(x["doc_id"], x["section"]) for x in address])

    def test_shipping_policy_switches_on_the_effective_date(self):
        for today, fee, threshold in (("2026-07-31", "₹99", "₹2,999"), ("2026-08-01", "₹149", "₹4,999")):
            with self.subTest(today=today):
                text = lambda query: " ".join(x["text"] for x in self.excerpts_at(today, query))
                self.assertIn(fee, text("What is the current shipping fee?"))
                self.assertIn(threshold, text("What is the free-shipping threshold?") + text("shipping fee"))
        # Once the update is in force it replaces the old fee and threshold; the Plus-member rule is not restated, so it stays.
        free = self.excerpts_at("2026-08-01", "Do you offer free shipping today?")
        sections = [(x["doc_id"], x["section"]) for x in free]
        self.assertNotIn(("shipping_policy", "§2.1"), sections)
        self.assertIn(("shipping_policy_update", "§1.2"), sections)
        self.assertIn(("shipping_policy", "§2.2"), sections)
        # On 31 July the notice is still a scheduled change: it is found only when the question is about the future.
        self.assertFalse(any(x["doc_id"] == "shipping_policy_update" for x in self.excerpts_at("2026-07-31", "What is the current shipping fee?")))
        later = self.excerpts_at("2026-07-31", "What will the shipping fee be from 1 August 2026?")
        self.assertTrue(any(x["doc_id"] == "shipping_policy_update" and x["future"] for x in later))

    def test_combined_shipping_and_address_question_cites_both_parts(self):
        owner, *_ = self.accounts()
        answer, route, calls, citations, *_ = self.chat(
            owner, "What is the current shipping fee, and can I change my delivery address?",
            script=[tool_call("search_policy", {"query": "shipping fee"}), tool_call("search_policy", {"query": "change delivery address"}, "call-2"),
                    AIMessage(content="Standard shipping costs ₹99 per order, free from ₹2,999 and for Plus members. You can change the address while the order is still processing.\n"
                                      "Sources: shipping_policy §2.1, shipping_policy §2.2, faq_general §5.1")])
        self.assertEqual("answer", route)
        self.assertEqual(["search_policy", "search_policy"], [c["name"] for c in calls])
        self.assertEqual([("shipping_policy", "§2.1"), ("shipping_policy", "§2.2"), ("faq_general", "§5.1")], [(c["doc_id"], c["section"]) for c in citations])

    # ---- bug 2: warranty ticket offer -> yes ---------------------------------------------------------
    def warranty_orders_by_customer(self):
        with self.db() as db:
            customers = [r[0] for r in db.execute("SELECT customer_id FROM customers ORDER BY customer_id")]
        found = {}
        for customer in customers:
            orders = build_tools(customer, lambda _: None, allow_proposals=False)[0].invoke({})["orders"]
            found[customer] = [o["order_id"] for o in orders if o["status"] == "delivered" and any(i["warranty_active"] for i in o["items"])]
        return found

    def http(self, customer, session, message, script=()):
        ScriptedChat.script = list(script)
        with patch.object(agent, "ChatOpenAI", ScriptedChat), patch.object(self.settings, "openai_api_key", "test-key"):
            response = TestClient(app).post("/chat", headers={"X-Customer-Id": customer}, json={"session_id": session, "message": message})
        return response.json()

    def warranty_offer(self, customer, session="warranty"):
        """The customer asks about a fault and the assistant offers a ticket (nothing is written)."""
        return self.http(customer, session, "My phone stopped charging. Is it covered?",
                         [tool_call("get_orders", {}), AIMessage(content="Yes, it is under warranty. Would you like me to open a warranty ticket?\nSources: none")])

    def tickets(self):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM support_tickets")]

    def single_warranty_customer(self):
        by_customer = self.warranty_orders_by_customer()
        customer = next(c for c, orders in by_customer.items() if len(orders) == 1)
        return customer, by_customer[customer][0], by_customer

    def test_yes_to_a_warranty_ticket_offer_creates_exactly_one_ticket(self):
        customer, order, _ = self.single_warranty_customer()
        offered = self.warranty_offer(customer)
        self.assertFalse(offered["pending_confirmation"])  # an offer, like a return offer, is not a prepared proposal
        self.assertEqual([], self.tickets())
        pending = SESSIONS["warranty"].pending
        self.assertEqual(("ticket", order, "warranty", True), (pending["kind"], pending["order_id"], pending["ticket_type"], pending["offered"]))
        done = self.http(customer, "warranty", "yes", [AIMessage(content="I've opened your warranty ticket.")])
        self.assertEqual([("create_support_ticket", "success")], [(c["name"], c["status"]) for c in done["tool_calls"]])
        rows = self.tickets()
        self.assertEqual(1, len(rows))
        self.assertEqual((customer, order, "warranty", "open", "My phone stopped charging. Is it covered?"),
                         (rows[0]["customer_id"], rows[0]["order_id"], rows[0]["ticket_type"], rows[0]["status"], rows[0]["summary"]))
        self.assertIn(rows[0]["ticket_id"], done["answer"])  # the real identifier is reported
        again = self.http(customer, "warranty", "yes")
        self.assertEqual(1, len(self.tickets()))
        self.assertEqual(done["answer"], again["answer"])

    def test_no_or_an_unrelated_message_after_a_warranty_offer_creates_no_ticket(self):
        customer, *_ = self.single_warranty_customer()
        self.warranty_offer(customer, "declined")
        self.http(customer, "declined", "no", [AIMessage(content="Okay, I won't open one.")])
        self.http(customer, "declined", "yes", [AIMessage(content="Nothing is waiting for confirmation.")])
        self.assertEqual([], self.tickets())
        self.warranty_offer(customer, "unrelated")
        unrelated = self.http(customer, "unrelated", "What is the current standard shipping fee?",
                              [tool_call("search_policy", {"query": "standard shipping fee"}), AIMessage(content="It is ₹99.\nSources: shipping_policy §2.1")])
        self.assertEqual("answer", unrelated["route"])
        self.assertIsNone(SESSIONS["unrelated"].pending)
        self.http(customer, "unrelated", "yes", [AIMessage(content="Nothing is waiting for confirmation.")])
        self.assertEqual([], self.tickets())

    def test_warranty_confirmation_is_revalidated_and_never_uses_a_foreign_order(self):
        customer, order, by_customer = self.single_warranty_customer()
        other = next(c for c in by_customer if c != customer)
        proposal = {"kind": "ticket", "order_id": order, "ticket_type": "warranty", "summary": "stopped working", "offered": True}
        refused = business_rules.create_support_ticket(other, order, "warranty", "stopped working")
        self.assertEqual("ORDER_NOT_FOUND", refused["error_code"])
        self.assertIsNone(agent._warranty_ticket_offer(other, "Would you like me to open a warranty ticket?", "stopped working", [order]))
        # The warranty ends before the customer says yes: the write is refused and the reply says so.
        with sqlite3.connect(self.db_path, factory=ClosingConnection) as db:
            db.execute("UPDATE shipments SET delivered_at='2020-01-01T10:00:00' WHERE order_id=?", (order,))
        failed = self.chat(customer, "yes", pending=proposal, script=[AIMessage(content="I couldn't open it.")])
        self.assertEqual([("create_support_ticket", "error")], [(c["name"], c["status"]) for c in failed[2]])
        self.assertEqual("WARRANTY_EXPIRED", failed[0].facts["error_code"])
        self.assertEqual([], self.tickets())

    def test_no_ticket_is_offered_when_the_order_is_ambiguous_or_has_no_warranty(self):
        by_customer = self.warranty_orders_by_customer()
        customer, orders = next((c, o) for c, o in by_customer.items() if len(o) > 1)
        offer = "Would you like me to open a warranty ticket?"
        self.assertIsNone(agent._warranty_ticket_offer(customer, offer, "it stopped working", orders))
        self.assertEqual(orders[0], agent._warranty_ticket_offer(customer, f"{orders[0]} is under warranty. {offer}", "it stopped working", orders)["order_id"])
        self.assertIsNone(agent._warranty_ticket_offer(customer, "Your warranty has ended.", "it stopped working", orders))
        self.assertIsNone(agent._warranty_ticket_offer(customer, offer, "it stopped working", []))

    def test_duplicate_warranty_ticket_is_refused_and_return_confirmation_is_unchanged(self):
        customer, order, _ = self.single_warranty_customer()
        self.assertTrue(business_rules.create_support_ticket(customer, order, "warranty", "stopped working")["created"])
        self.assertEqual("ALREADY_REQUESTED", business_rules.create_support_ticket(customer, order, "warranty", "stopped working")["error_code"])
        self.assertEqual(1, len(self.tickets()))
        buyer, return_order, item = self.returnable_item()
        proposed = self.chat(buyer, f"Please return {item} from {return_order}, it is defective", script=[AIMessage(content="Prepared.")])
        self.assertTrue(proposed[4])
        self.assertEqual("return", proposed[5]["kind"])
        done = self.chat(buyer, "yes", pending=proposed[5], script=[AIMessage(content="Saved.")])
        self.assertEqual([("create_return_request", "success")], [(c["name"], c["status"]) for c in done[2]])
        self.assertEqual((1, 1), (self.count("return_requests"), self.count("support_tickets")))

    # ---- bug 3: a refused return is a successful check -----------------------------------------------
    def test_expired_return_is_denied_with_its_reason_and_the_check_is_logged_as_success(self):
        customer, order, item = self.returnable_item()
        with sqlite3.connect(self.db_path, factory=ClosingConnection) as db:
            db.execute("UPDATE shipments SET delivered_at='2026-01-01T10:00:00' WHERE order_id=?", (order,))
        result = self.chat(customer, f"Can I return {item} from {order}? It is defective.", script=[AIMessage(content="No.")])
        checks = [c for c in result[2] if c["name"] == "check_return_eligibility"]
        self.assertEqual(["success"], [c["status"] for c in checks])
        self.assertEqual("RETURN_WINDOW_EXPIRED", result[0].facts["error_code"])
        self.assertFalse(result[0].facts["can_be_returned"])
        self.assertEqual([{"doc_id": "returns_policy", "section": "§2.1"}], result[3])
        self.assertEqual(0, self.count("return_requests"))

    def test_eligible_return_still_goes_through_confirmation(self):
        customer, order, item = self.returnable_item()
        result = self.chat(customer, f"Please return {item} from {order}, it is defective", script=[AIMessage(content="Prepared.")])
        self.assertEqual([("get_orders", "success"), ("search_policy", "success"), ("check_return_eligibility", "success"), ("create_return_request", "pending_confirmation")],
                         [(c["name"], c["status"]) for c in result[2]])
        self.assertTrue(result[4])
        self.assertEqual(0, self.count("return_requests"))

    def test_real_failures_and_foreign_orders_are_still_errors(self):
        self.assertEqual("success", agent._check_status({"eligible": True}))
        for code in ("RETURN_WINDOW_EXPIRED", "NON_RETURNABLE", "NOT_DELIVERED", "OPENED_NOT_RETURNABLE"):
            self.assertEqual("success", agent._check_status({"eligible": False, "error_code": code}), code)
        for result in ({"error": "TOOL_ERROR"}, {"error_code": "ORDER_NOT_FOUND"}, {"error_code": "INVALID_ITEM"},
                       {"error_code": "INVALID_REASON"}, {"error_code": "POLICY_PARAMETER_INVALID"}):
            self.assertEqual("error", agent._check_status(result), result)
        customer, order, item = self.returnable_item()
        args = {"order_id": order, "item_ids": [item], "reason": "defective"}
        with patch("app.tools.support_tools.business_check_return", side_effect=RuntimeError("database is locked")):
            crashed = self.chat(customer, "Is that order eligible for a refund?", script=[tool_call("check_return_eligibility", args), AIMessage(content="Sorry, something failed.")])
        self.assertEqual([("check_return_eligibility", "error")], [(c["name"], c["status"]) for c in crashed[2]])
        other = next(c for c, *_ in [self.accounts()[2:3]])
        foreign = build_tools(other, lambda _: None, allow_proposals=False)[2].invoke(args)
        self.assertEqual(("ORDER_NOT_FOUND", "error"), (foreign["error_code"], agent._check_status(foreign)))
        reply = self.chat(other, f"Can I return {item} from {order}?")
        self.assertEqual((NOT_FOUND, "no_answer", "error"), (str(reply[0]), reply[1], reply[2][0]["status"]))

    # ---- bug 4: spacing when an inline citation is removed -------------------------------------------
    def test_removing_an_inline_citation_keeps_the_words_apart(self):
        clean = agent._remove_inline_policy_citations
        self.assertEqual("It changes from 1 August 2026.", clean("It changes from(shipping_policy_update §1.1)1 August 2026."))
        self.assertEqual("It is not covered. I can’t open a ticket.", clean("It is not covered.(warranty_policy §3.1)I can’t open a ticket."))
        self.assertEqual("It is not covered. I can’t open a ticket.", clean("It is not covered. (warranty_policy §3.1) I can’t open a ticket."))
        self.assertEqual("Standard shipping costs ₹99.", clean("Standard shipping costs ₹99 (shipping_policy §2.1)."))
        self.assertEqual("No citation here.", clean("No citation here."))

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
