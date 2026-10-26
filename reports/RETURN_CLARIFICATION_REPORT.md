# ResolveBot return clarification and OpenAI runtime retest

Date: 2026-10-25  
Configured model: `gpt-6-luna`  
Live-test budget limit: USD 0.05  
Live tests used a temporary copy of the SQLite database.

## Root cause

The provider rejected a request to `POST /v1/responses` with HTTP 400, `param=input`, `type=invalid_request_error`, and the message `No tool output found for function call <call_id>.` The diagnostic captured the complete provider error without recording or exposing credentials.

The model returned one assistant message containing nine parallel `check_return_eligibility` calls. The graph had room to execute only four more calls under its six-call limit. It appended tool outputs for those four calls but omitted outputs for the other five, then sent the incomplete assistant/tool message history back to OpenAI. The Responses API requires one output for every function call in that message, so it rejected the next request. This happened both in a fresh session and after a previous policy answer; conversation history was not the cause.

The failing user-facing response before the fix was:

> I couldn't complete that request right now. Please try again or contact ShopSphere support. (OpenAIInvalidRequestError)

## Changes made

- `app/agents/support_agent.py:161` resolves return references against only the authenticated customer’s database orders. Broad or ambiguous references list matching account items and ask which one; a uniquely identified item is retained across turns. It asks for a reason before eligibility evaluation and uses the Python eligibility tool. Eligibility questions do not create proposals; a clear action request can create a pending proposal.
- `app/agents/support_agent.py:270` now emits a `ToolMessage` for every generated function call. Calls beyond the tool budget receive a safe tool-limit result, and the graph returns a clarification instead of submitting an invalid conversation to OpenAI.
- `tests/test_scenarios.py:400` and `:421` cover broad return clarification and complete tool-call output when the call limit is reached.
- `tests/run_return_clarification_live.py:118` exercises the live HTTP and OpenAI paths using a temporary database, including defective reasons, ambiguity, confirmation, duplicate confirmation, and customer isolation.
- `scripts/chat_terminal.py:131` prints a blank line between terminal exchanges, as requested earlier.

## Results

The exact query `can i return my products?` returned HTTP 200 with route `clarify`, a successful `get_orders` call, account item choices, and a request to identify the item and reason. It made no LLM or eligibility calls and created no proposal. The same result occurred in a fresh session and after a policy-history turn. This avoids asking the customer to supply an order ID and avoids checking every item for a broad question.

Live HTTP conversation checks, all against the temporary database:

| Scenario | Result |
|---|---|
| Broad return question, fresh and after history | Both clarified; HTTP 200; no eligibility check or write |
| Unique speaker selection | Clarified for reason; no eligibility check until reason was supplied |
| Changed-mind reason | Python eligibility check succeeded; no write while the customer was only asking eligibility |
| Explicit “Please create the return” | Eligibility rechecked; proposal pending; database row count stayed 0 |
| “Yes, go ahead” | HTTP 200; exactly one return row inserted in temporary DB |
| Duplicate confirmation | No second write; row count remained 1 |
| Reuse session as another customer | HTTP 403; no database change |
| Defective eligibility question | Asked for reason first; then ran Python eligibility; no proposal or write |
| Multiple speaker matches | Listed matching choices and clarified; item ID selection then reason were handled in-session; eligibility was evaluated and rejected for the selected database item; no write |
| Required response fields | Present on all HTTP 200 responses |

The live runner completed 16 HTTP checks with expected statuses (15 HTTP 200 and the expected cross-customer HTTP 403). The slowest response was the policy-history call at 4,275.8 ms, under the 30-second limit. All other recorded HTTP responses were under 52 ms. The original database still had zero return requests and zero support tickets after testing.

The deterministic suite also passed: 17 tests, 0 failures.

## Actual provider usage

Only the policy-history question reached OpenAI in this retest; the exact return query and business-rule flow are handled deterministically. Two real `gpt-6-luna` calls occurred:

| Customer query / call | Input tokens | Output tokens | Total tokens | Cached input | Reasoning | Provider call duration |
|---|---:|---:|---:|---:|---:|---:|
| Return-window history, call 1 | 530 | 29 | 559 | 0 | 0 | 1,956.7 ms |
| Return-window history, call 2 | 960 | 71 | 1,031 | 0 | 0 | 2,089.1 ms |
| **Suite total** | **1,490** | **100** | **1,590** | **0** | **0** | — |

The provider reported all token fields; no missing tokens were estimated. Mocked calls are excluded. Cached and reasoning values above are the provider-reported values. HTTP elapsed time for that customer query, including both calls and the API/tool flow, was 4,275.8 ms.

## Verification limits

The live return flow did not require the model to infer item identity, reason, eligibility, or confirmation. That work is intentionally deterministic and database-backed. The live test did make real OpenAI requests for the history seed. The nine-call provider failure was reproduced during diagnosis; after the fix, the exact return request no longer produces the hazardous parallel-call request, and a deterministic graph test verifies that every generated call receives an output.
