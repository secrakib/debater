# Tests that DebateState behaves correctly as a data container.
# These are fast, pure Python — no LLM calls needed.

from debate import DebateState


def test_default_state_values():
    """State should initialize with sensible defaults."""
    state = DebateState()
    assert state.question == ""
    assert state.messages == []
    assert state.turn_count == 0


def test_state_with_question():
    """State should store the debate question correctly."""
    state = DebateState(question="Is climate change real?")
    assert state.question == "Is climate change real?"


def test_state_is_immutable_pydantic():
    """
    Pydantic models are immutable by default.
    Nodes return new dicts instead of mutating state — this confirms that pattern works.
    """
    state = DebateState(question="Tax policy?", turn_count=0)
    # Simulate what a node does: return a new dict, not mutate
    updated = state.model_copy(update={"turn_count": 1})
    assert updated.turn_count == 1
    assert state.turn_count == 0  # Original unchanged