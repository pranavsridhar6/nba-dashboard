import json, re
from langchain_core.messages import ToolMessage

NUM = re.compile(r"(?<![\w.])-?\$?\.?\d+(?:,\d{3})*(?:\.\d+)?%?M?")

def answer_numbers(text: str) -> list[str]:
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)    # drop dates like 2026-04-19
    text = re.sub(r"\b\d{4}-\d{2}\b", "", text)          # then seasons like 2025-26
    return NUM.findall(text)

def source_numbers(messages) -> list[float]:
    found = []
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                data = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                continue
            for row in data.get("rows", []):
                for v in row.values():
                    if isinstance(v, (int, float)) and v == v:
                        found.append(float(v))
    return found
   
    

def parse(token: str) -> tuple[float, float]:
    """'$57.7M' -> (57700000.0, 50000.0): value and half of the last shown digit."""
    t = token.replace("$", "").replace(",", "").replace("%", "")
    millions = t.endswith("M")
    t = t.rstrip("M")
    decimals = len(t.split(".")[1]) if "." in t else 0
    value, tol = float(t), 0.5 * 10 ** -decimals
    if millions:
        value, tol = value * 1e6, tol * 1e6
    return value, tol

def grounded(token: str, sources: list[float]) -> bool:
    value, tol = parse(token)
    if value == int(value) and 0 <= value <= 10:
        return True                      # small counts in prose ("top 5")
    for s in sources:
        for scale in (1, 100, 0.01):     # 0.683 <-> 68.3%
            if abs(value - s * scale) <= tol:
                return True
    return False


def ungrounded(answer: str, messages) -> list[str]:
    sources = source_numbers(messages)
    return [t for t in answer_numbers(answer) if not grounded(t, sources)]

if __name__ == "__main__":
    from langchain_core.messages import ToolMessage
    tool = ToolMessage(content=json.dumps({"rows": [
        {"wins": 56, "losses": 26, "win_pct": 0.683, "salary": 57736350}]}),
        tool_call_id="x")
    print(ungrounded("BOS went 56-26 (.683), paid $57.7M.", [tool]))   # expect []
    print(ungrounded("BOS went 56-26, 68.3% wins, top 5.", [tool]))    # expect []
    print(ungrounded("BOS went 58-24 in 2025-26.", [tool]))           # expect ['58', '24']