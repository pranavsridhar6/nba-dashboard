from langchain_core.messages import ToolMessage, AIMessage
import json
from agent.graph import check

tool = ToolMessage(content=json.dumps({"rows": [{"wins": 56, "losses": 26}]}), tool_call_id="x")
bad  = AIMessage(content="Boston went 56-26 and has won 18 titles.")

r1 = check({"messages": [tool, bad], "retries": 0})
print(r1["verdict"], r1["retries"], r1["messages"][0].content)

r2 = check({"messages": [tool, bad], "retries": 1})
print(r2["verdict"], r2["messages"][0].content[-60:])