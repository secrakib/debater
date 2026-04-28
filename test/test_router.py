# Tests the routing decisions: who speaks next, when to stop.
# No LLM needed — router is pure conditional logic.

from debate import router, DebateState
from langchain_core.messages import AIMessage


def test_router_goes_to_democrat_after_republican(one_turn_state):
    """After republican speaks, router should send to democrat."""
    # one_turn_state fixture comes from conftest.py automatically
    result = router(one_turn_state)
    assert result == "democrat"


def test_router_goes_to_republican_when_no_messages(empty_state):
    """With no messages yet, router should start with republican."""
    result = router(empty_state)
    assert result == "republican"


def test_router_ends_when_max_turns_reached(full_turn_state):
    """
    When turn_count >= MAX_TURNS (1), debate should end.
    full_turn_state has turn_count=1 which equals MAX_TURNS.
    """
    result = router(full_turn_state)
    assert result == "end"


def test_router_continues_if_under_max_turns(one_turn_state):
    """turn_count=0 is below MAX_TURNS=1, so debate should continue."""
    result = router(one_turn_state)
    assert result != "end"