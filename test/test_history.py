# Tests the core message-reframing logic.
# This is the trickiest part of your code — deserves thorough testing.

from langchain_core.messages import AIMessage, HumanMessage
from debate import build_history


def test_own_messages_become_ai_messages():
    """
    When republican calls build_history("republican"),
    its own messages should come back as AIMessage (I said this).
    """
    messages = [AIMessage(content="Cut taxes!", name="republican")]
    history = build_history(messages, "republican")

    assert len(history) == 1
    assert isinstance(history[0], AIMessage)
    assert history[0].content == "Cut taxes!"


def test_opponent_messages_become_human_messages():
    """
    When republican calls build_history("republican"),
    democrat's messages should become HumanMessage (someone spoke to me).
    """
    messages = [AIMessage(content="Raise taxes!", name="democrat")]
    history = build_history(messages, "republican")

    assert len(history) == 1
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "Raise taxes!"


def test_mixed_conversation_history():
    """
    Alternating messages should be correctly classified from each speaker's perspective.
    republican sees: own=AI, opponent=Human
    """
    messages = [
        AIMessage(content="Cut taxes.", name="republican"),
        AIMessage(content="Raise taxes.", name="democrat"),
        AIMessage(content="Cut them more.", name="republican"),
    ]
    history = build_history(messages, "republican")

    assert isinstance(history[0], AIMessage)    # republican's own
    assert isinstance(history[1], HumanMessage) # democrat's — seen as opponent
    assert isinstance(history[2], AIMessage)    # republican's own again


def test_empty_messages_returns_empty_history():
    """Edge case: no messages yet should return an empty list."""
    history = build_history([], "republican")
    assert history == []