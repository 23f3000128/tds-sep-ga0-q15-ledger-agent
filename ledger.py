"""Deterministic query engine over the Acme ledger export.

Rules from the API root:
  - Business dates use Asia/Kolkata.
  - An order id can appear more than once; the row with the latest updated_at is current.
  - Only orders with status 'paid' count as revenue.
Money is converted to USD with /rates (usd_per_unit) and reported to the cent.
"""
import gzip, json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
USD_PER_UNIT = {"USD": 1.0, "EUR": 1.09, "INR": 0.01106}

HERE = Path(__file__).parent


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load(path=HERE / "export.ndjson.gz"):
    raw = gzip.decompress(Path(path).read_bytes()).decode()
    rows = [json.loads(l) for l in raw.strip().splitlines() if l.strip()]
    latest = {}
    for r in rows:
        cur = latest.get(r["id"])
        if cur is None or _ts(r["updated_at"]) > _ts(cur["updated_at"]):
            latest[r["id"]] = r
    out = []
    for r in latest.values():
        r = dict(r)
        r["usd"] = r["amount"] * USD_PER_UNIT[r["currency"]]
        c = _ts(r["created_at"]).astimezone(IST)
        u = _ts(r["updated_at"]).astimezone(IST)
        r["date"] = c.date().isoformat()          # business date (IST) of creation
        r["month"] = c.strftime("%Y-%m")
        r["quarter"] = f"{c.year}-Q{(c.month - 1) // 3 + 1}"
        r["updated_date"] = u.date().isoformat()
        r["weekday"] = c.strftime("%A")
        out.append(r)
    return out


ORDERS = load()
VALUES = {
    "region": sorted({r["region"] for r in ORDERS}),
    "product": sorted({r["product"] for r in ORDERS}),
    "status": sorted({r["status"] for r in ORDERS}),
    "currency": sorted({r["currency"] for r in ORDERS}),
    "customer_example": sorted({r["customer"] for r in ORDERS})[:3],
    "date_range": [min(r["date"] for r in ORDERS), max(r["date"] for r in ORDERS)],
}

GROUP_KEYS = {"region", "product", "customer", "month", "quarter", "currency", "status", "date", "weekday", "id"}


def _match(r, f):
    for k in ("region", "product", "customer", "currency", "status", "id", "month", "quarter", "weekday"):
        v = f.get(k)
        if v:
            vals = v if isinstance(v, list) else [v]
            if r[k].lower() not in [str(x).lower() for x in vals]:
                return False
    dfield = "updated_date" if f.get("date_field") == "updated_at" else "date"
    if f.get("start_date") and r[dfield] < f["start_date"]:
        return False
    if f.get("end_date") and r[dfield] > f["end_date"]:
        return False
    if f.get("min_qty") is not None and r["qty"] < f["min_qty"]:
        return False
    return True


def _measure(rows, measure):
    if measure == "sum_usd":
        return sum(r["usd"] for r in rows)
    if measure == "count":
        return len(rows)
    if measure == "sum_qty":
        return sum(r["qty"] for r in rows)
    if measure == "distinct_customers":
        return len({r["customer"] for r in rows})
    if measure == "avg_usd":
        return sum(r["usd"] for r in rows) / len(rows) if rows else 0
    if measure == "avg_qty":
        return sum(r["qty"] for r in rows) / len(rows) if rows else 0
    if measure == "max_usd":
        return max((r["usd"] for r in rows), default=0)
    if measure == "min_usd":
        return min((r["usd"] for r in rows), default=0)
    raise ValueError(f"unknown measure {measure}")


def _fmt(measure, v, ratio=False, percent=False):
    if ratio:
        return round(v, 2) if percent else round(v, 4)
    if measure in ("count", "sum_qty", "distinct_customers"):
        return int(v)
    return round(v + 0.0, 2)


def run(q):
    """q = {
      measure: sum_usd|count|sum_qty|distinct_customers|avg_usd|avg_qty|max_usd|min_usd,
      filters: {region, product, customer, currency, status, id, month(YYYY-MM), quarter(YYYY-Qn),
                weekday, start_date, end_date (IST YYYY-MM-DD inclusive), date_field: created_at|updated_at, min_qty},
      denominator: optional {measure, filters}  -> ratio (x100 if percent)
      percent: bool,
      group_by: optional key, order: desc|asc, rank: 1-based, top_n: int,
      return: value|key|key_and_value|list_keys|list_pairs
    }"""
    f = q.get("filters") or {}
    rows = [r for r in ORDERS if _match(r, f)]
    measure = q.get("measure", "sum_usd")
    gb = q.get("group_by")

    def value_of(rs):
        v = _measure(rs, measure)
        d = q.get("denominator")
        if d:
            drows = [r for r in ORDERS if _match(r, d.get("filters") or {})]
            if gb:  # same group restriction applied by caller
                drows = [r for r in drows if r[gb] == rs[0][gb]] if rs else []
            dv = _measure(drows, d.get("measure", measure))
            v = (v / dv) if dv else 0
            if q.get("percent"):
                v *= 100
        return v

    if not gb:
        return _fmt(measure, value_of(rows), bool(q.get("denominator")), bool(q.get("percent")))

    groups = defaultdict(list)
    for r in rows:
        groups[r[gb]].append(r)
    scored = [(k, value_of(v)) for k, v in groups.items()]
    rev = q.get("order", "desc") != "asc"
    scored.sort(key=lambda kv: (-kv[1] if rev else kv[1], kv[0]))
    ret = q.get("return", "key")
    pct = bool(q.get("denominator"))
    pp = bool(q.get("percent"))
    if q.get("top_n"):
        top = scored[: int(q["top_n"])]
        if ret == "list_pairs":
            return [{"key": k, "value": _fmt(measure, v, pct, pp)} for k, v in top]
        return [k for k, _ in top]
    rank = int(q.get("rank") or 1)
    if rank > len(scored):
        return None
    k, v = scored[rank - 1]
    if ret == "value":
        return _fmt(measure, v, pct, pp)
    if ret == "key_and_value":
        return {"key": k, "value": _fmt(measure, v, pct, pp)}
    return k
