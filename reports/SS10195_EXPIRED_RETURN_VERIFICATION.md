# SS-10195 expired-return verification

`APP_TODAY` is `2026-07-01`; SQLite’s `return_window_days` is 30. C1001’s item SS-10195-1 (Orbitra Screen Guard X5) belongs to delivered order SS-10195, delivered January 27, 2026. It was 155 days after delivery at `APP_TODAY`.

The earlier successful confirmation test used **SS-10255**, item SS-10255-1 (Voltix 30W Car Charger), delivered June 10, 2026. That item was 21 days after delivery, inside the 30-day window. The test’s pending proposal and one-row write were performed only on a temporary database copy. It did not use SS-10195.

I tested SS-10195 separately through live `/chat` using a temporary database copy:

| Request | Tool calls | Pending | Temp DB return rows |
|---|---|---:|---:|
| Eligibility question with explicit `changed_mind` | `get_orders` success; `check_return_eligibility` error (`RETURN_WINDOW_EXPIRED`) | false | 0 |
| Explicit create-return request with the same item/reason | `get_orders` success; `check_return_eligibility` error (`RETURN_WINDOW_EXPIRED`); no create tool call | false | 0 |

Both HTTP requests returned 200 with a customer-facing expired-window answer. The original SQLite database remained unchanged at zero return requests and zero support tickets. No application fix was needed.

Reproduction harness: `tests/run_ss10195_expired_return_live.py`.
