# Integration test: runs the FULL compiled graph with a real (or mocked) LLM.
# This is the slowest test — run it less frequently (e.g., before deployment).
# We still mock the LLM here to keep CI fast and avoid API costs.

from unittest.mock import patch, MagicMock
from debate import build_graph, DebateState


def make_fake_llm_response(text: str):
    mock = MagicMock()
    mock.content = text
    return mock


@patch("debate.llm")
def test_graph_runs_one_full_turn(mock_llm):
    """
    Full graph integration test:
    - Republican speaks → Democrat speaks → graph ends (MAX_TURNS=1)
    - Verifies the graph is wired correctly end-to-end.
    """
    # LLM alternates responses for republican and democrat
    mock_llm.invoke.side_effect = [
        make_fake_llm_response("We should cut spending."),   # republican
        make_fake_llm_response("We need more investment."),  # democrat
    ]

    graph = build_graph()
    final_state = graph.invoke(DebateState(question="What is the best economic policy?"))

    # After 1 full turn, there should be exactly 2 messages
    assert len(final_state["messages"]) == 2
    assert final_state["messages"][0].name == "republican"
    assert final_state["messages"][1].name == "democrat"

    # turn_count should be 1 (one full republican+democrat cycle)
    assert final_state["turn_count"] == 1