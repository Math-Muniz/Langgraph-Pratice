import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2)

# --- Lógica do Mongo (Short-Memory) ---
MONGO_URI = os.getenv("MONGO_URI")
with MongoDBSaver.from_conn_string(os.getenv(MONGO_URI)) as checkpointer:

    def call_model(state: MessagesState):
        """Chama o modelo e retorna a resposta DENTRO DE UMA LISTA."""
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    
    graph = builder.compile(checkpointer=checkpointer)

    # --- Lógica do Chat Interativo  ---

    thread_id = "test_1" 
    print(f"Iniciando chat com memória. ID da Conversa: {thread_id}")
    print("Digite 'sair' a qualquer momento para terminar o chat.")

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ["sair", "exit"]:
            print("🤖 Até mais!")
            break

        messages_to_send = [{"role": "user", "content": user_input}]

        print("🤖 IA: ", end="", flush=True)
        
        final_response = None
        for chunk in graph.stream(
            {"messages": messages_to_send},
            config,
        ):
            if "call_model" in chunk:
                final_response = chunk["call_model"]["messages"][-1]
        
        if final_response:
            print(final_response.content)