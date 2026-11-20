from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from agent.tools import sql_query

load_dotenv()
llm = ChatAnthropic(model="claude-sonnet-5-5").bind_tools([sql_query])

from agent.prompt import SYSTEM_PROMPT
SYSTEM = SystemMessage(content=SYSTEM_PROMPT)

def agent(state: State):
    reply = llm.invoke([SYSTEM] + state["messages"])
    return {"messages": [reply]}

from typing import Literal

def route(state: State) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return "__end__"

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode

builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([sql_query]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route)
builder.add_edge("tools", "agent")
graph = builder.compile()

if __name__ == "__main__":
    q = "What is Jaylen Brown's RAPM?"
    out = graph.invoke({"messages": [("user", q)]}, {"recursion_limit": 10})
    for m in out["messages"]:
        m.pretty_print()