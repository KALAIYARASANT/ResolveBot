# ResolveBot Token Usage Report

**Final recorded suite date:** 2026-10-25  
**Configured model:** `gpt-6-luna`  
**Live budget:** `$0.05` (`RESOLVEBOT_LIVE_TEST_BUDGET_USD`, configurable; default `$0.05`)  
**Final suite outcome:** 20 actual model calls across 13 successful customer messages; 13,233 input tokens, 720 output tokens, 13,953 total tokens; measured estimated cost `$0.00158241`. Budget was not exhausted. No provider token fields were missing.

## Measurement method

The test harness instrumented the real `ChatOpenAI` instance with a LangChain callback and sent conversations to the running FastAPI/LangGraph service over HTTP. Token fields came from provider response usage metadata for each completed model invocation. Deterministic tests, HTTP validation probes, session rejection, clarification handled in Python, and direct confirmation turns were not counted as model calls.

OpenAI's official GPT-6 Luna page lists Standard rates per 1M tokens of `$0.10` uncached input, `$0.01` cached input, and `$0.50` output ([official model pricing](https://developers.openai.com/api/docs/models/gpt-6-luna)). The suite used the default Standard tier and had no long-context requests. Per-call cost is calculated as:

`((input_tokens - cached_input_tokens) x $0.10 + cached_input_tokens x $0.01 + output_tokens x $0.50) / 1,000,000`

Provider-reported reasoning tokens are shown separately and are already included in output tokens; they are not added again. No token values are estimated. All final-suite calls returned input/output/total usage. One call reported 1,121 cached input tokens; the other calls reported zero cached input tokens. All reported reasoning-token values were zero.

## Usage by customer query and test scenario

A customer message that did not invoke a model is shown with zero tokens/cost. HTTP duration is the complete request/response duration; LLM duration is the sum of callback-measured model-call durations for that query.

| Query/scenario | Model calls | Input | Output | Total | Cached input | Reasoning | LLM duration | HTTP response | Estimated cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `order_tracking` | 2 | 1157 | 77 | 1234 | 0 | 0 | 3233.1 ms | 3559.8 ms | $0.00015420 |
| `foreign_order_lookup` | 2 | 1016 | 54 | 1070 | 0 | 0 | 3005.3 ms | 3050.9 ms | $0.00012860 |
| `current_shipping_fee` | 2 | 1082 | 55 | 1137 | 0 | 0 | 2480.0 ms | 2537.6 ms | $0.00013570 |
| `missing_order_id` | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 ms | 42.2 ms | $0.00000000 |
| `return_proposal` | 3 | 1801 | 144 | 1945 | 0 | 0 | 4546.0 ms | 4601.1 ms | $0.00025210 |
| `return_details_changed` | 4 | 3630 | 150 | 3780 | 1121 | 0 | 5646.7 ms | 5716.0 ms | $0.00033711 |
| `return_confirmation` | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 ms | 33.2 ms | $0.00000000 |
| `duplicate_return_confirmation` | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 ms | 5.3 ms | $0.00000000 |
| `warranty_proposal` | 3 | 2087 | 115 | 2202 | 0 | 0 | 5130.1 ms | 5176.5 ms | $0.00026620 |
| `warranty_confirmation` | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 ms | 35.1 ms | $0.00000000 |
| `cancellation_proposal` | 3 | 1981 | 112 | 2093 | 0 | 0 | 6026.4 ms | 6092.4 ms | $0.00025410 |
| `cancellation_confirmation` | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 ms | 18.0 ms | $0.00000000 |
| `unrelated_query` | 1 | 479 | 13 | 492 | 0 | 0 | 1028.3 ms | 1095.5 ms | $0.00005440 |

## Per-model-call provider usage

| # | Customer query / scenario | Model | Input | Output | Total | Cached input | Reasoning | Model-call duration | Estimated cost |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `order_tracking` | `gpt-6-luna` | 478 | 26 | 504 | 0 | 0 | 1737.3 ms | $0.00006080 |
| 2 | `order_tracking` | `gpt-6-luna` | 679 | 51 | 730 | 0 | 0 | 1495.8 ms | $0.00009340 |
| 3 | `foreign_order_lookup` | `gpt-6-luna` | 478 | 26 | 504 | 0 | 0 | 1583.4 ms | $0.00006080 |
| 4 | `foreign_order_lookup` | `gpt-6-luna` | 538 | 28 | 566 | 0 | 0 | 1421.9 ms | $0.00006780 |
| 5 | `current_shipping_fee` | `gpt-6-luna` | 477 | 20 | 497 | 0 | 0 | 1189.9 ms | $0.00005770 |
| 6 | `current_shipping_fee` | `gpt-6-luna` | 605 | 35 | 640 | 0 | 0 | 1290.1 ms | $0.00007800 |
| 7 | `return_proposal` | `gpt-6-luna` | 500 | 49 | 549 | 0 | 0 | 1333.1 ms | $0.00007450 |
| 8 | `return_proposal` | `gpt-6-luna` | 586 | 47 | 633 | 0 | 0 | 1437.9 ms | $0.00008210 |
| 9 | `return_proposal` | `gpt-6-luna` | 715 | 48 | 763 | 0 | 0 | 1775.0 ms | $0.00009550 |
| 10 | `return_details_changed` | `gpt-6-luna` | 565 | 26 | 591 | 0 | 0 | 1340.3 ms | $0.00006950 |
| 11 | `return_details_changed` | `gpt-6-luna` | 931 | 42 | 973 | 0 | 0 | 1298.7 ms | $0.00011410 |
| 12 | `return_details_changed` | `gpt-6-luna` | 1010 | 40 | 1050 | 0 | 0 | 1338.2 ms | $0.00012100 |
| 13 | `return_details_changed` | `gpt-6-luna` | 1124 | 42 | 1166 | 1121 | 0 | 1669.5 ms | $0.00003251 |
| 14 | `warranty_proposal` | `gpt-6-luna` | 488 | 26 | 514 | 0 | 0 | 1457.9 ms | $0.00006180 |
| 15 | `warranty_proposal` | `gpt-6-luna` | 741 | 41 | 782 | 0 | 0 | 1870.5 ms | $0.00009460 |
| 16 | `warranty_proposal` | `gpt-6-luna` | 858 | 48 | 906 | 0 | 0 | 1801.7 ms | $0.00010980 |
| 17 | `cancellation_proposal` | `gpt-6-luna` | 486 | 26 | 512 | 0 | 0 | 1210.1 ms | $0.00006160 |
| 18 | `cancellation_proposal` | `gpt-6-luna` | 687 | 43 | 730 | 0 | 0 | 3456.8 ms | $0.00009020 |
| 19 | `cancellation_proposal` | `gpt-6-luna` | 808 | 43 | 851 | 0 | 0 | 1359.5 ms | $0.00010230 |
| 20 | `unrelated_query` | `gpt-6-luna` | 479 | 13 | 492 | 0 | 0 | 1028.3 ms | $0.00005440 |

## Suite aggregate

| Measure | Provider-reported / calculated value |
|---|---:|
| Model | `gpt-6-luna` (all calls) |
| Actual model calls | 20 |
| Successful customer messages, including confirmations/clarification | 13 |
| Customer messages that invoked the model | 8 |
| Input tokens | 13,233 |
| Output tokens | 720 |
| Total tokens | 13,953 |
| Cached input tokens | 1,121 |
| Reasoning tokens (subset of output; not added again) | 0 |
| Estimated Standard API cost | $0.00158241 |
| Average total tokens per successful customer message | 1073.31 |
| Average total tokens per model-invoking customer message | 1744.12 |
| Configured budget / remaining after this final suite | $0.05 / $0.04841759 |
| Missing provider token rows | 0 |

Confirmation messages, the repeated confirmation, and missing-order clarification each produced a real HTTP response but no model call. The token averages include zero-token messages as described; the model-invoking average excludes those. API validation probes (400/401) and the rejected 403 session request are excluded from customer-query averages because they do not invoke the model.

## Accounting scope

These totals cover the **final instrumented suite run** written to the report. Earlier exploratory live runs used during defect diagnosis were not retained as a final suite artifact and are not included in these totals. No missing or unretained token figures are inferred or added. This report does not expose the API key or raw prompts/identifiers.
