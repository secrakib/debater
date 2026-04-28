# Tests republican_node and democrat_node WITHOUT making real LLM API calls.
# We "mock" the LLM so tests are fast, free, and deterministic.
# unittest.mock is built into Python — no extra install needed.

from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from debate import republican_node, democrat_node, DebateState


def make_fake_llm_response(text: str):
    """
    Helper that creates a fake LLM response object.
    MagicMock lets us fake any object — here we fake what llm.invoke() returns.
    """
    mock_response = MagicMock()
    mock_response.content = text
    return mock_response


# --- Republican Node ---

@patch("debate.llm")  # Replaces the real `llm` in debate.py with a fake during this test
def test_republican_node_first_turn(mock_llm, empty_state):
    """Republican should produce an AIMessage with name='republican' on first turn."""
    mock_llm.invoke.return_value = make_fake_llm_response("We must cut spending.")

    result = republican_node(empty_state)

    # Node returns a dict with updated messages
    assert "messages" in result
    last_message = result["messages"][-1]
    assert last_message.name == "republican"
    assert last_message.content == "We must cut spending."


@patch("debate.llm")
def test_republican_node_calls_llm_once(mock_llm, empty_state):
    """LLM should be called exactly once per node invocation."""
    mock_llm.invoke.return_value = make_fake_llm_response("Lower taxes work.")

    republican_node(empty_state)

    mock_llm.invoke.assert_called_once()  # Verifies no accidental double-calls


@patch("debate.llm")
def test_republican_node_does_not_increment_turn_count(mock_llm, empty_state):
    """
    Republican node intentionally doesn't increment turn_count.
    Only democrat closes a full turn. Verify this design is preserved.
    """
    mock_llm.invoke.return_value = make_fake_llm_response("Cut taxes.")

    result = republican_node(empty_state)

    # turn_count key should NOT be in the returned dict
    assert "turn_count" not in result


# --- Democrat Node ---

@patch("debate.llm")
def test_democrat_node_increments_turn_count(mock_llm, one_turn_state):
    """Democrat node should increment turn_count to signal a full turn completed."""
    mock_llm.invoke.return_value = make_fake_llm_response("We need public investment.")

    result = democrat_node(one_turn_state)

    assert result["turn_count"] == one_turn_state.turn_count + 1


@patch("debate.llm")
def test_democrat_node_message_name(mock_llm, one_turn_state):
    """Democrat's message should have name='democrat'."""
    mock_llm.invoke.return_value = make_fake_llm_response("Tax the wealthy.")

    result = democrat_node(one_turn_state)

    last_message = result["messages"][-1]
    assert last_message.name == "democrat"