from ast import Add
from dotenv import load_dotenv
import os
from typing import Sequence, TypedDict, List, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: Union[int, float], b: Union[int, float]):
    """
    Add two numbers together. Use this tool whenever you need to calculate, sum, total, plus, or answer a math question involving addition. Use for any math or calculation question involving +.
    """
    print(f"[TOOL CALLED] {a} + {b} = {a + b}")
    return a + b

@tool
def multipy(a: Union[int, float], b: Union[int, float]):
    """
    Multiplies two number togathere. Use when you need to multipy number even easy ones
    """
    print(f"[TOOL CALLED] {a} * {b} = {a * b}")
    return a * b

tools = [add, multipy]

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model_name="openai/gpt-4-turbo-preview"
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    sysem_prompt = SystemMessage(content="You are a scientific AI assistant. Always use available tools when a user asks for math, calculation, or data you can retrieve via a tool. Do not answer math questions yourself—use the 'add' tool for all addition or sum questions.")
    response = llm.invoke([sysem_prompt] + state["messages"])
    print("AI:", response.content)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()


history = []
user_input = input("Enter: ")

while True:
    history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": history})
    history = result["messages"]
    user_input = input("Enter: ")
