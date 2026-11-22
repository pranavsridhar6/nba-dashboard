# 1. imports
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from agent.tools import sql_query
from agent.prompt import SYSTEM_PROMPT
from agent.grounding import ungrounded
from langgraph.checkpoint.memory import MemorySaver



# 2. state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    retries: int
    verdict: str

# 3. model + prompt
load_dotenv()
llm = ChatAnthropic(model="claude-sonnet-5-5").bind_tools([sql_query])
SYSTEM = SystemMessage(content=SYSTEM_PROMPT)

# 4. nodes
def text_of(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))

def agent(state: State):
    reply = llm.invoke([SYSTEM] + state["messages"])
    return {"messages": [reply]}

def check(state: State):
    answer = text_of(state["messages"][-1])
    bad = ungrounded(answer, state["messages"])
    if not bad:
        return {"verdict": "ok"}
    if state.get("retries", 0) < 1:
        fix = HumanMessage(content=(
            f"These numbers are not in any query result: {bad}. "
            "Re-query to get them, or remove them. Answer again."))
        return {"messages": [fix], "retries": state.get("retries", 0) + 1, "verdict": "retry"}
    note = AIMessage(content=answer + f"\n\nUnverified numbers: {bad}. Treat them with caution.")
    return {"messages": [note], "verdict": "flagged"}

# 5. routers
def route(state: State) -> Literal["tools", "check"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "check"

def route_check(state: State) -> Literal["agent", "__end__"]:
    if state.get("verdict") == "retry":
        return "agent"
    return "__end__"

# 6. wiring
builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([sql_query]))
builder.add_node("check", check)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route)
builder.add_edge("tools", "agent")
builder.add_conditional_edges("check", route_check)
graph = builder.compile(checkpointer=MemorySaver())

# 7. run
if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "demo"}, "recursion_limit": 15}
    for q in ["What was Boston's 2025-26 regular season record?",
              "And their playoff record that season?"]:
        out = graph.invoke({"messages": [("user", q)], "retries": 0}, cfg)
        print(f"\nQ: {q}\nA: {text_of(out['messages'][-1])}")