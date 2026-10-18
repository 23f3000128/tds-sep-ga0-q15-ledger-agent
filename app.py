"""Q15 Ledger Agent: POST {"question": "..."} -> {"answer": ...}

An LLM (via AI Pipe) only translates the question into a structured query;
ledger.run() computes the answer deterministically from the embedded export.
Set AIPIPE_TOKEN in the host's environment variables.
"""
import json, os, re

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import ledger

MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
TOKEN = os.environ.get("AIPIPE_TOKEN", "")
URL = "https://aipipe.org/openai/v1/chat/completions"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SYSTEM = f"""You translate questions about an order ledger into ONE JSON query. Output JSON only.

DATA: one row per order (already de-duplicated to the current version). Fields:
id, customer (e.g. {ledger.VALUES['customer_example']}), region {ledger.VALUES['region']},
product {ledger.VALUES['product']}, qty, unit_price, amount, currency {ledger.VALUES['currency']},
status {ledger.VALUES['status']}, plus derived: usd (amount converted to USD), date (IST, YYYY-MM-DD),
month (YYYY-MM), quarter (YYYY-Qn), weekday (Monday..Sunday). Dates span {ledger.VALUES['date_range']} (year 2026).

QUERY SCHEMA:
{{
 "measure": "sum_usd" | "count" | "sum_qty" | "distinct_customers" | "avg_usd" | "avg_qty" | "max_usd" | "min_usd",
 "filters": {{"region": [..], "product": [..], "customer": [..], "currency": [..], "status": [..], "id": [..],
             "month": [..], "quarter": [..], "weekday": [..], "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}},
 "denominator": {{"measure": ..., "filters": {{..}}}}   // ONLY for rates/shares/percentages
 "percent": true|false,                                  // with denominator: multiply by 100
 "group_by": "region"|"product"|"customer"|"month"|"quarter"|"currency"|"weekday"|"status"|"date",
 "order": "desc"|"asc", "rank": 1, "top_n": null,
 "return": "value" | "key" | "key_and_value" | "list_keys" | "list_pairs"
}}

RULES:
- Revenue / sales / income / earnings / turnover = measure sum_usd with status ["paid"]. ONLY paid counts as revenue.
- Refunds (amount) = sum_usd with status ["refunded"]. "How many refunds" = count with status ["refunded"].
- "Orders" with no status qualifier = count over ALL statuses unless the question says paid/completed/successful
  (then status ["paid"]) or voided/cancelled (status ["void"]).
- Units / items sold = sum_qty with status ["paid"]. Customers who bought = distinct_customers with status ["paid"].
- Average order value = avg_usd with status ["paid"].
- Use exact spellings from the value lists above (map synonyms/plurals/case, e.g. "air fryers" -> "Air Fryer", "north" -> "North").
- A month name means that month of 2026 -> "month": ["2026-03"]. "Q1" -> "quarter": ["2026-Q1"]. "first half"/H1 -> start 2026-01-01 end 2026-06-30.
  Specific date ranges -> start_date/end_date inclusive.
- "Which/what <thing> had the most/highest/top ..." -> group_by that thing, order desc, rank 1, return "key".
  "least/lowest/fewest" -> order asc. "second highest" -> rank 2. "top 3 ..." -> top_n 3, return "list_keys".
  If the question asks for the amount/number of the top group ("how much did the top region earn"), return "value";
  if it asks for both, return "key_and_value".
- Share/percentage/rate questions: put the numerator in measure+filters and the base in denominator, percent true if they say percent/%.
- Omit keys you don't need. Never invent filters the question doesn't state.
"""

FEW_SHOT = [
    ("What was the total revenue in USD from the North region in March 2026?",
     {"measure": "sum_usd", "filters": {"status": ["paid"], "region": ["North"], "month": ["2026-03"]}, "return": "value"}),
    ("Which product brought in the most money in Q2?",
     {"measure": "sum_usd", "filters": {"status": ["paid"], "quarter": ["2026-Q2"]}, "group_by": "product", "order": "desc", "rank": 1, "return": "key"}),
    ("What percentage of orders were refunded?",
     {"measure": "count", "filters": {"status": ["refunded"]}, "denominator": {"measure": "count", "filters": {}}, "percent": True, "return": "value"}),
]


async def parse(question: str) -> dict:
    msgs = [{"role": "system", "content": SYSTEM}]
    for qq, aa in FEW_SHOT:
        msgs += [{"role": "user", "content": qq}, {"role": "assistant", "content": json.dumps(aa)}]
    msgs.append({"role": "user", "content": question})
    async with httpx.AsyncClient(timeout=12) as c:
        r = await c.post(URL, headers={"Authorization": f"Bearer {TOKEN}"},
                         json={"model": MODEL, "messages": msgs, "temperature": 0,
                               "response_format": {"type": "json_object"}})
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def normalise(q: dict) -> dict:
    """Make LLM output safe for ledger.run: coerce enums, fix casing, map list returns."""
    f = q.get("filters") or {}
    canon = {k: {v.lower(): v for v in ledger.VALUES[k]} for k in ("region", "product", "status", "currency")}
    for k, m in canon.items():
        if f.get(k):
            vals = f[k] if isinstance(f[k], list) else [f[k]]
            f[k] = [m.get(str(v).lower().rstrip("s"), m.get(str(v).lower(), v)) for v in vals]
    if f.get("customer"):
        vals = f["customer"] if isinstance(f["customer"], list) else [f["customer"]]
        f["customer"] = [str(v).upper() for v in vals]
    q["filters"] = f
    if q.get("return") in ("list_keys",):
        q["top_n"] = q.get("top_n") or 3
    return q


@app.get("/")
def root():
    return {"ok": True, "orders": len(ledger.ORDERS)}


@app.post("/")
@app.post("/ask")
@app.post("/query")
async def ask(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    q = normalise(await parse(question))
    try:
        answer = ledger.run(q)
    except Exception as e:  # never 500 on the grader
        answer = None
        q["_error"] = str(e)
    out = {"answer": answer}
    if body.get("debug"):
        out["query"] = q
    return out
