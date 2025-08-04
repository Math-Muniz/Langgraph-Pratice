import os
from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import START, END, StateGraph 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from colorama import Fore
from langgraph.prebuilt import ToolNode
from tool import simple_screener
# Load Environment Variables
load_dotenv()

# Create a LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# Create a Tool
tools = [simple_screener]

# Bind LLM wwith Tools
llm_with_tools = llm.bind_tools(tools)

# Create a ToolNode
tool_node = ToolNode(tools)

# Create a State
class State(dict):
    messages: Annotated[list, add_messages]

# Build LLM node

def chatbot(state: State):
    print(state['messages'])
    return {"messages":[llm_with_tools.invoke(state['messages'])]}

# Create a Router
def router(state:State): 
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls: 
        return "tools" 
    else: 
        return END 

# Assemble Graph

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", router)

# Add Memory

memory = InMemorySaver() 
graph = graph_builder.compile(checkpointer=memory)

# Run
if __name__ == '__main__': 
    while True: 
        prompt = input("🤖 Pass your prompt here: " )
        result = graph.invoke({"messages":[{"role":"user", "content":prompt}]}, config={"configurable":{"thread_id":1234}})
        print(Fore.LIGHTYELLOW_EX + result['messages'][-1].content + Fore.RESET) 