"""Scratch: find where exceptions and hard-cap status appear in saved ShamSports pages. Delete after."""
import re

KEYWORDS = ["exception", "hard cap", "hard-cap", "apron", "remaining"]

for team in ["PHI", "BOS", "DET"]:
    html = open(f"data/contracts/raw/{team}.html", encoding="utf-8").read()
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", html))
    print(f"\n========== {team} ==========")
    seen = set()
    for kw in KEYWORDS:
        for m in re.finditer(re.escape(kw), text, re.IGNORECASE):
            start = max(0, m.start() - 120)
            snippet = text[start:m.end() + 160]
            key = snippet[:60]
            if key in seen:
                continue
            seen.add(key)
            print(f"[{kw}] ...{snippet}...")
            if len(seen) >= 12:
                break