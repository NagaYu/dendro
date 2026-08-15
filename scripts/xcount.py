"""X (Twitter) weighted character count.

X weights most Latin/punctuation at 1 and CJK at 2, capping at 280 — so a
Japanese post effectively gets ~140 characters. URLs always count as 23
regardless of their real length (t.co wrapping).
"""
import re, sys

LIGHT = [(0, 4351), (8192, 8205), (8208, 8223), (8242, 8247)]
URL = re.compile(r"https?://\S+")

def weighted(text: str) -> int:
    text = URL.sub("x" * 23, text)
    n = 0
    for ch in text:
        cp = ord(ch)
        n += 1 if any(lo <= cp <= hi for lo, hi in LIGHT) else 2
    return n

def report(label, posts):
    print(f"--- {label} ---")
    ok = True
    for i, p in enumerate(posts, 1):
        w = weighted(p)
        flag = "OK " if w <= 280 else "OVER"
        if w > 280: ok = False
        print(f"  {i}/{len(posts)}  {w:3d}/280  {flag}")
    print(f"  => {'all within limit' if ok else 'SOME OVER LIMIT'}\n")
    return ok
