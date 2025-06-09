from re import S
from dotenv import load_dotenv
import os
from typing import Sequence, TypedDict, List, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# RAG агент с которым можно поговорить о гармоничесокм анализе

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model_name="openai/gpt-4-turbo-preview"
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pdf_path = "Матанализ.pdf"
collection_name = "Matan"

pdf_loader = PyPDFLoader(pdf_path)                # may error as well as line below
pages = pdf_loader.load()
print(f"Loadeed {len(pages)} pages fo PDF document")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1300,
    chunk_overlap = 300
)

pages_split = text_splitter.split_documents(pages)

chroma_dir = os.getenv("CHROMA_DIR")
try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name = collection_name
    )
    print(f"Created ChromaDB vector store!")
except Exception as e:
    print("Error setting up ChromaDB", e)
    raise


retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 4}     # retruns 4 best chunks
) 


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information on harmonical analysis form file stored in vector embedding format
    """

    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information"
    result = ""
    for i in range(len(docs)):
        result += f"Document {i}:\n {docs[i].page_content}"
    return result

tools = [retriever_tool]

llm = llm.bind_tools(tools)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    return "end"

system_prompt = """ You are an Agent that helps people anderstand math theoremes and descriptions. You have acess to
database on harmonical analysis. If question might be relevant - you ask database and then explain answer. DO not
think for yourslefe, try to use as much only data from database when answering math questions. Please always cyte cpecific
part of documents you are provided in database
"""

def call_llm(state: AgentState):
    """
    Function to call LLM with its state
    """
    message = llm.invoke([SystemMessage(system_prompt)] + state["messages"])
    print("AI:\n" + message.content)
    return {"messages": [message]}    # actually adds it

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        # Find the tool in the tools list by name
        tool = next((tool for tool in tools if tool.name == t['name']), None)
        if tool is None:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tool.invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever", take_action)
graph.add_edge("retriever", "llm")
graph.set_entry_point("llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue": "retriever",
         "end": END
    }
)

app = graph.compile()

#//show graph
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import io

# img_bytes = app.get_graph().draw_mermaid_png()
# img = mpimg.imread(io.BytesIO(img_bytes), format='png')
# plt.imshow(img)
# plt.axis('off')
# plt.show()
#// show graph


history = []

while True:
    user_input = input("Enter: ")
    history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": history})
    history = result["messages"]