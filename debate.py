from typing import Literal, Annotated
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

MAX_TURNS = 3  # One turn = republican + democrat

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# --- State ---
class DebateState(BaseModel):
    question: str = ""
    messages: list = []
    turn_count: int = 0
    next_node: str = "republican"  # controls flow


# --- Nodes ---
def input_node(state: DebateState):
    print(f"\n[Input] Question: {state.question}")
    return state


def republican_node(state: DebateState):
    system = SystemMessage(content=(
        "You are a Republican politician. Debate the given topic strictly from a "
        "conservative Republican perspective. Be concise (2-3 sentences)."
    ))
    history = [HumanMessage(content=state.question)] + state.messages
    response = llm.invoke([system] + history)
    new_message = AIMessage(content=response.content, name="republican")
    
    print(f"\n{new_message}")
    
    return {"messages": state.messages + [new_message],
        "next_node": "democrat"}


def democrat_node(state: DebateState):
    system = SystemMessage(content=(
        "You are a Democratic politician. Debate the given topic strictly from a "
        "progressive Democratic perspective. Be concise (2-3 sentences)."
    ))
    history = [HumanMessage(content=state.question)] + state.messages
    response = llm.invoke([system] + history)
    
    new_message = AIMessage(content= response.content,name="democrat")
    print(f"\n{new_message}")
    
    return {"messages": state.messages + [new_message], 
            "turn_count": state.turn_count + 1,
            "next_node": "republican" }


def router(state: DebateState) -> str:
    if state.turn_count >= MAX_TURNS:
        print("\n[Router] Max turns reached. Ending debate.")
        return "end"
    return state.next_node


# --- Build Graph ---
def build_graph():
    graph = StateGraph(DebateState)

    graph.add_node("input", input_node)
    graph.add_node("republican", republican_node)
    
    graph.add_node("democrat", democrat_node)

    graph.set_entry_point("input")
    graph.add_edge("input", "republican")

    graph.add_conditional_edges(
        "republican",
        router,
        {"democrat": "democrat", "end": END}
    )

    graph.add_conditional_edges(
        "democrat",
        router,
        {"republican": "republican", "end": END}
    )
    

    return graph.compile()

graph = build_graph()

#graph.invoke({'question':'is refugee good for country?'})

