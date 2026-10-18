# /// script
# dependencies = ["httpx"]
# ///
"""Usage: uv run test_ledger.py https://YOUR-APP.onrender.com/
Fires all questions at once (like the grader) and compares with precomputed answers."""
import asyncio, json, sys, time
import httpx
URL = sys.argv[1]
CASES = json.load(open("expected.json"))

def same(a, b):
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        return abs(a - b) < 0.006
    return str(a).strip().lower() == str(b).strip().lower()

async def one(c, case):
    t = time.time()
    r = await c.post(URL, json={"question": case["question"], "debug": True}, timeout=30)
    j = r.json()
    ok = same(j.get("answer"), case["expected"])
    print(f"{'PASS' if ok else 'FAIL'} {time.time()-t:5.1f}s  got={j.get('answer')!r} expected={case['expected']!r}\n      {case['question']}")
    if not ok:
        print("      parsed:", json.dumps(j.get("query")))
    return ok

async def main():
    async with httpx.AsyncClient() as c:
        res = await asyncio.gather(*[one(c, x) for x in CASES])
    print(f"\n{sum(res)}/{len(res)} passed")
asyncio.run(main())
