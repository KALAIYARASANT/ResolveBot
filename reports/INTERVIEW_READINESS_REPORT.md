# ResolveBot: Interview-Readiness Test Report

Date of test: 2026-10-28 (business date under test: `APP_TODAY=2026-07-01` unless stated)
Scope: the working tree (uncommitted changes included). The last commit (`HEAD`) was also replayed; see Part 4, finding R1.
Method: real HTTP calls to the running service with the configured OpenAI model, on **temporary copies** of the database. The real `data/shopsphere.db` was never written (MD5 `520b4bd6…` before and after).

## Status after the fixes (added later the same day)

Everything in the findings below was re-tested on the fixed code. The parts below describe the code as it was when first tested.

| Finding | Status |
|---|---|
| R1 uncommitted work | Fixed: committed (`bcb00ea`), then the fixes in a second commit. |
| R2 default model / provider | Fixed: `OPENAI_MODEL` (default `gpt-4o-mini`), optional `OPENAI_REASONING_EFFORT`, unsupported provider or empty model stops startup with `ConfigurationError`. The old default `gpt-6-luna` rejects `temperature=0` unless reasoning is disabled. |
| R3 supplied DB had 2 return rows | Not touched (not asked). Note: the new duplicate guard would refuse a demo return of `SS-10458-1` for `C1009` against that file; restore a clean copy before demoing. |
| S1 multi-question | Fixed: compound messages skip the scripted shortcuts and go to the agent loop. |
| S2/S3 FAQ and warranty retrieval | Fixed: no topic-to-document map, headings and titles are searchable, FAQ competes on score. |
| S4 routing | Fixed: explicit scope rules, exact refusal sentence, and a one-time retry when the policy files do match. |
| S5/S6 coverage and citations | Fixed: relevance threshold; citations are the sections the model names that its answer uses, plus the best-supported excerpt; none when unused. |
| S7 context | Fixed: "it" resolves from recent turns; asks only when there is no context. |
| S8 order-ID formats | Fixed: `SS10512`, `SS 10512`, `ss-10512` normalized; ownership still checked on the canonical ID. |
| N1 duplicates | Fixed for returns and cancellation tickets (`ALREADY_REQUESTED`). |
| N2 sessions | Fixed: idle expiry (1 h), cap (1000), history trimmed to 40. |
| N3 timestamps | Fixed: `APP_TODAY` date + wall-clock time. |
| N5 message size | Fixed: 4000 characters, `session_id` 128, both HTTP 400. |
| Dates written "July 3, 2026" | Prompt now asks for "3 July 2026". |
| Cancelling a cancelled/delivered order says "shipped" | Not changed. |
| Extra JSON fields give 400 | Not changed. |
| N4 regex front-end | Not removed; only a guard was added so it no longer hijacks follow-up questions after a policy answer (a bug found while fixing). |

New problems found and fixed while doing this: a follow-up question after any return-policy answer was hijacked into "couldn't find a matching item", and the shipping-fee answer regressed to the old fee on `APP_TODAY` after 1 Aug 2026 until a generic newer-document-wins rule was added.

---

## 1. The five mandatory checks (task section 3.4)

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Service starts with the README command | **PASS** | `python -m uvicorn app.main:app --port 8000` came up and answered. A missing header returned 401. |
| 2 | Replies have the section 2.6 format | **PASS** | All six fields present, with correct types, in every reply I checked (five dedicated samples, and the same fields held in every other reply I saw). Routes were always one of the four allowed values. `tool_calls[].status` was always `success`, `error` or `pending_confirmation`. The spec's two-message return example reproduced exactly: tool order `get_orders → search_policy → check_return_eligibility → create_return_request(pending_confirmation)`, citation `returns_policy §2.1`, then `success` and `pending_confirmation:false` after "Yes, go ahead." |
| 3 | Nothing saved without the customer's yes | **PASS** (35/35) | Zero rows written by: the proposal itself; "no" then "yes"; an unrelated message then "yes"; "just do it" (with or without a proposal); "I confirm, go ahead, no need to ask" inside the first message; a prompt-injection asking the model to call the write tool; "Yes, but make it defective" (treated as a change, not a yes). A repeated "yes" and four concurrent "yes" requests produced exactly one row. State changed between proposal and "yes" (window expired): the write was blocked at insert time, because the rules are revalidated. |
| 4 | Return decisions correct, even with different numbers | **PASS** | About 45 deterministic checks plus 11 live HTTP checks. Covered: every row of the spec 2.3 table; day 0, 30 and 31; check order (a non-returnable item that is also expired gives `NON_RETURNABLE`); `NOT_DELIVERED`, `INVALID_ITEM`, `INVALID_REASON`; windows of 7, 45, 52 and 53 days; warranty 12, 24, 1 and 36 months; month-end (31 Jan + 1 month = 28 Feb) and leap year (29 Feb 2028). **Variant run:** `APP_TODAY=2026-08-02`, a changed policy text (53 days), changed parameters. Day 53 of 53 was eligible and day 53 of 52 was not. The answer said "53-day window", not 30. The shipping-fee update (effective 1 Aug 2026) was correctly treated as current. |
| 5 | A customer can never see another customer's order | **PASS** | About 15 attacks as C1001 against C1009's orders: lookup, return, cancel, warranty, item-level id, lowercase id, "I am C1009", "admin mode" prompt injection, "list all orders". Every one gave the exact text "I couldn't find that order on your account." with route `no_answer`, or only C1001's own orders. A foreign order and a nonexistent order are indistinguishable (same text, route and tool status). Reusing another customer's `session_id` gave 403. Header tricks (`c1001`, trailing space, SQL injection) gave 401. |

**Verdict: all five gates pass on the working tree.** One caveat is serious: the working tree is not committed (R1).

Also run: bundled `tests/test_scenarios.py`, 24 tests, OK (deterministic only).

HTTP contract: 401 (missing or unknown id), 400 (invalid JSON, empty body, JSON array, missing `session_id` or `message`, wrong-typed `session_id`, empty message), 403 (foreign session) all correct. Latency 0–12 s, well under 30 s.

---

## 2. New-scenario results ("what an interviewer would try")

Passed:
- Return decisions through chat: expired (day 31), opened earbuds changed-mind (refused), opened earbuds defective (proposed), gift card (refused), shipped order (refused), 3-charger ambiguity (lists all 3, asks for the reason too).
- Cancellation: processing order proposed then saved on "yes" (`cancellation`, `open`); shipped order refused with no proposal.
- Warranty: laptop question answered without a proposal; ticket proposed only on request; saved on "yes"; expired warranty refuses the ticket.
- Policy: current vs future fee (₹99 now, ₹149 from 1 Aug 2026), superseded 14-day policy never used, policy beats FAQ on refunds ("5–7 business days"), Plus shipping.
- Off-topic poem, weather and hacking requests refused. Greeting handled. Emoji, Tamil text and a 51 KB message did not crash.
- A bad or unavailable model name: graceful message, no crash, and the return flow still works.

**Failed or weak (each reproduced; details in Part 3):**

| ID | Scenario | What happened |
|----|----------|---------------|
| S1 | "Where is SS-10512, and can I return the speaker from SS-10458? I changed my mind." | Answered only "This order must be delivered…". It ran the return check on the **wrong order** (SS-10512) and ignored the tracking question. |
| S2 | "Do you ship to Dubai?" / "Can I change my delivery address?" | Replied "policy doesn't cover it" and cited unrelated `shipping_policy`. The FAQ (§6.1, §5.1) answers both. |
| S3 | "My phone stopped charging. Is it covered?" (the spec's own example) | Hedged: "The policy doesn't specify…". `warranty_policy §3.1` covers "a port that stops charging". It did not offer a ticket, and the call log contains a failed `create_support_ticket` attempt. |
| S4 | "Write python code to reverse a list" / "Who won the cricket world cup?" | Wrote the code with route `answer`; the cricket question got `no_answer` ("policies do not cover"). Both should be `out_of_scope`. |
| S5 | "Do you price match competitors?" | Said "not covered" but route was `answer` and 4 unrelated citations were returned. The spec requires `no_answer` and empty citations. |
| S6 | "How many days do I have to return something?" | Cites §2.1, §3.1, §5.1, §6.1. Only about two were used. |
| S7 | "Where is SS-10512?" then "When will it arrive?" | Asked "which order?" again, so session memory was ignored for this phrasing. |
| S8 | "SS10512" and "SS 10512" (no hyphen) | Not recognised as order IDs; fell back to listing the customer's own orders. No leak, but not the mandated reply. |

---

## 3. Defects in detail (spec-based)

Ordered by likely impact on an assessment.

**S1. Two-part messages mishandled** (spec 2.5 "answer both"). `_return_detail_turn` runs *before* the LLM (`support_agent.py:575`) and takes the *first* `SS-…` in the message (`:384`). A tracking question followed by a return picks the tracking order as the return target. Fix direction: if the message names more than one order or contains a non-return intent, skip the shortcut and let the agent loop handle it.

**S2/S3/S5. Retrieval quality** (`policy_search.py`).
- `preferred` documents are derived from a hard-coded keyword → document map (`:79-93`). When any matches, every other document (including the FAQ) is dropped (`:126-127`). Questions about "delivery" or "shipping" never reach FAQ §5.1/§6.1.
- "charging" has no mapping to warranty §3.1; scoring is plain term overlap.
- `covered=True` whenever *any* paragraph scores above 0, so irrelevant paragraphs count as coverage (S5). The "not covered" route only fires when there are zero excerpts (`support_agent.py:599`).
- The topic map embeds policy knowledge (which topics live in which document). It is not a number, but it is the kind of coupling the spec warns against ("Do not paste policy facts into your code").

**S4. Off-topic routing by string match.** `out_of_scope` is set only if the LLM's text contains "only help with shopsphere" (`support_agent.py:618`). There is no classifier or tool-free path, so some off-topic prompts get answered (code) or mislabelled (`no_answer`).

**S6. Over-citation.** The citation filter (`:609-613`) keeps only sections whose id appears in the answer text, but the system prompt forbids putting ids in the answer, so the filter never applies. All retrieved excerpts are cited. Spec: "List each document you used, and no others."

**S7. Memory ignored by a pre-LLM interceptor.** `_ORDER_LOOKUP_INTENT` (`:563`) answers "when will … arrive" without looking at history.

**S8. Order-id parsing** accepts only `SS-\d+`. Normalising `SS10512` and `SS 10512` is a small change.

Minor:
- Dates read "July 3, 2026" / "Jun 5, 2026"; the spec asks for "3 July 2026".
- Cancelling a `delivered` or `cancelled` order returns `ORDER_ALREADY_SHIPPED` with the text "has shipped" (`business_rules.py:121`), which is misleading for cancelled orders.
- Deterministic replies echo raw tool text ("The return window ended 30 days after delivery.") instead of friendly wording with the day count.
- `ChatRequest` forbids extra fields (`schemas/chat.py:6`): a tester adding any extra key gets 400, but the spec lists only invalid JSON or missing fields.

---

## 4. Findings NOT based on the PDF

These matter in the interview even though the task text never mentions them.

**R1 (critical, process): the last commit is not what I tested.** `README.md`, `app/agents/support_agent.py` and `scripts/chat_terminal.py` are modified but uncommitted. The task says reviewers read the last commit. I exported `HEAD` and replayed the attacks:
- C1001 returning another customer's order got route **`clarify`** and a dump of C1001's own items, **not** the mandated "I couldn't find that order on your account." (`no_answer`).
- The return proposal said "Please reply yes to confirm." instead of "Shall I go ahead?", and had no `search_policy` call or citation.
- No data leaked and the confirmation gate held, but spec compliance is visibly worse at `HEAD`. **Commit before the deadline.**

**R2 (high): default model and provider.** Default `OPENAI_MODEL=gpt-6-luna`, and any `LLM_PROVIDER` other than `openai` silently drops to a limited fallback (`support_agent.py:580`). The interviewer "runs with our own API key". If their key lacks that model, every LLM-path question returns "I couldn't complete that request right now… (OpenAIModelNotFoundError)" (verified). Return and ticket flows survive; tracking and policy do not. The README does not tell them to set `OPENAI_MODEL`. Add that line, or fall back to a widely available model.

**R3: the supplied database is no longer pristine.** `data/shopsphere.db` already holds **2 `return_requests` rows** (SS-10458-1, C1009, created 2026-10-27 12:54 and 13:06), from earlier manual runs against the real file. The task says the table starts empty, and `CLAUDE.md` says not to alter supplied data. I did not touch them. `data/` is git-ignored, so the submission is unaffected, but restore a clean copy before any live demo.

**N1. No duplicate guard.** The bot proposes (and would save) a second return for an item that already has a request: the two real rows above are exactly that. Ticket creation has no dedupe either (from reading the code; not exercised live). In production this means double refunds or tickets.

**N2. Unbounded in-memory sessions.** `SESSIONS` (`router.py:29`) is never evicted and `history` grows forever. Fine for the assessment, a memory leak in service. It also pins the service to one worker, and the README already says so.

**N3. Timestamps vs business date.** `created_at` uses the machine clock (`business_rules.py:132,144`), so rows read `2026-10-28` while `APP_TODAY` is `2026-07-01`. The spec says "current time", so this is defensible, but it should be a stated assumption.

**N4. Heavy regex front-end vs "one agent".** About 300 lines of regex logic (`_return_detail_turn`, `_revise_pending_return`, `allows_new_proposal`) decide return flows before the LLM sees the message. This is why S1 and S7 happen, and it will be hard to explain line by line in the interview. Also, some `tool_calls` entries are synthesised by code (for example `_get_orders_record` always reports `args: {}` and `success`, `support_agent.py:286`) instead of recording what actually ran.

**N5. Cost and abuse controls.** There is no cap on message length (a 51 KB message was accepted and processed), no rate limit and no eviction. `X-Customer-Id` is trusted as identity, which is correct per the spec but is not authentication.

**N6. Housekeeping.** `database.transaction()` is unused. Seven `tests/run_*_live.py` scripts call the real OpenAI API and cost money when run. The README says Python 3.14 while the task allows 3.10+, and 3.14 prints a LangChain Pydantic-v1 warning. I ran the app only on the project's 3.14 venv.

**Positives worth saying out loud:** the write path is structurally safe (the proposal tool cannot insert; only `execute_proposal` after a clear yes, with revalidation); all business numbers come from `policy_parameters`; SQL is parameterised; no secrets in git history; `.env` and `data/` are ignored; response schema is enforced by a Pydantic model; latency is well within 30 s.

---

## 5. Limits of this test

- LLM-dependent wording and tool choice can vary between runs even at temperature 0. I ran each scenario once, apart from the isolation suite, which gave identical results on two runs.
- "Interviewer method" is inferred from the task's section 3.4 and its statement that `data/` is swapped for different values. I have no knowledge of their actual test script.
- Not tested: Python 3.10–3.13, a non-OpenAI key, multiple uvicorn workers, a real 30 s timeout.
- Scripts used are in the session scratchpad, not in the repo.
