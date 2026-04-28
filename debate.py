from typing import Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

MAX_TURNS = 1  # One turn = republican + democrat

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# --- State ---
class DebateState(BaseModel):
    question: str = ""
    messages: list = []
    turn_count: int = 0


# --- Key fix: build a tailored history for each agent ---
def build_history(messages: list, speaker_name: str) -> list:
    """
    Each agent needs its own perspective on the conversation.
    - Messages they sent themselves  → AIMessage   (I said this)
    - Messages the opponent sent     → HumanMessage (someone spoke to me)

    Without this, the LLM sees a pile of generic AIMessages with no clear
    sense of who is talking to whom, causing incoherent or repetitive debate.
    """
    history = []
    for msg in messages:
        if msg.name == speaker_name:
            # This agent's own prior statements — keep as AIMessage
            history.append(AIMessage(content=msg.content))
        else:
            # The opponent's statements — reframe as HumanMessage so the LLM
            # treats them as "the other party speaking to me"
            history.append(HumanMessage(content=msg.content))
    return history


# --- Nodes ---
def input_node(state: DebateState):
    print(f"\n[Input] Question: {state.question}")
    return state


def republican_node(state: DebateState):
    system = SystemMessage(content=(
        "You are a Republican politician debating a Democrat. Respond to the topic "
        "and directly counter your opponent's last argument if one exists. "
        "Be concise (2-3 sentences). Conservative perspective only."
    ))

    # First turn: only the question. Subsequent turns: full tailored history.
    if not state.messages:
        history = [HumanMessage(content=state.question)]
    else:
        # Question sets the context once at the top, then the back-and-forth follows
        history = [HumanMessage(content=state.question)] + build_history(state.messages, "republican")

    response = llm.invoke([system] + history)
    new_message = AIMessage(content=response.content, name="republican")

    print(f"\n[Republican]: {new_message.content}")

    return {
        "messages": state.messages + [new_message]  # turn_count intentionally NOT incremented here — only democrat closes a full turn
    }


def democrat_node(state: DebateState):
    system = SystemMessage(content=(
        "You are a Democratic politician debating a Republican. Respond to the topic "
        "and directly counter your opponent's last argument. "
        "Be concise (2-3 sentences). Progressive Democratic perspective only."
    ))

    history = [HumanMessage(content=state.question)] + build_history(state.messages, "democrat")

    response = llm.invoke([system] + history)
    new_message = AIMessage(content=response.content, name="democrat")

    print(f"\n[Democrat]: {new_message.content}")

    return {
        "messages": state.messages + [new_message],
        "turn_count": state.turn_count + 1   # A full turn is republican + democrat
    }


def router(state: DebateState) -> Literal["republican", "democrat", "end"]:
    if state.turn_count >= MAX_TURNS:
        print("\n[Router] Max turns reached. Ending debate.")
        return "end"

    # Determine whose turn it is based on the last speaker
    if state.messages and state.messages[-1].name == "republican":
        return "democrat"
    return "republican"


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
