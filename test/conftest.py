# conftest.py is special in pytest — fixtures defined here are
# automatically available to ALL test files without importing them.

import pytest
from langchain_core.messages import AIMessage
from debate import DebateState


@pytest.fixture
def empty_state():
    """A fresh debate state with no messages — used for first-turn tests."""
    return DebateState(question="Should taxes be raised?", messages=[], turn_count=0)


@pytest.fixture
def one_turn_state():
    """
    A state after the Republican has spoken once.
    Used to test: Democrat response, router after republican speaks.
    """
    republican_msg = AIMessage(content="We should cut taxes.", name="republican")
    return DebateState(
        question="Should taxes be raised?",
        messages=[republican_msg],
        turn_count=0
    )


@pytest.fixture
def full_turn_state():
    """
    A state after both Republican AND Democrat have spoken once (1 full turn).
    Used to test: router stopping at MAX_TURNS, turn counting.
    """
    republican_msg = AIMessage(content="We should cut taxes.", name="republican")
    democrat_msg = AIMessage(content="We need investment in public services.", name="democrat")
    return DebateState(
        question="Should taxes be raised?",
        messages=[republican_msg, democrat_msg],
        turn_count=1  # 1 full turn = republican + democrat
    )