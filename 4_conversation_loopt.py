from dotenv import load_dotenv
import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
llm = ChatOpenAI(                     #   actually using OpenRouterAPI
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct-v0.2"
)

def process(state: AgentState) -> AgentState:
    """this node will answer question"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(response.content))
    print(response.content)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.set_finish_point("process")

agent = graph.compile()

history = []
user_input = input("Enter: ")

while True:
    history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": history})
    history = result["messages"]
    user_input = input("Enter: ")