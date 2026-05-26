"""Tests for the LangGraph pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from scholar_rag.graph import ResearchState, _route_after_router, build_graph


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

class TestRouteAfterRouter:
    def test_routes_direct_to_responder(self):
        state = ResearchState(
            query="", intent="direct", retrieved_texts=[],
            agent_scratchpad=[], answer="", sources=[],
        )
        assert _route_after_router(state) == "responder"

    def test_routes_retrieval_to_retriever(self):
        state = ResearchState(
            query="", intent="retrieval", retrieved_texts=[],
            agent_scratchpad=[], answer="", sources=[],
        )
        assert _route_after_router(state) == "retriever"

    def test_routes_out_of_scope_to_refuser(self):
        state = ResearchState(
            query="", intent="out_of_scope", retrieved_texts=[],
            agent_scratchpad=[], answer="", sources=[],
        )
        assert _route_after_router(state) == "refuser"

    def test_unknown_intent_routes_to_retriever(self):
        state = ResearchState(
            query="", intent="", retrieved_texts=[],
            agent_scratchpad=[], answer="", sources=[],
        )
        assert _route_after_router(state) == "retriever"


# ---------------------------------------------------------------------------
# Router node
# ---------------------------------------------------------------------------

class TestRouterNode:
    @patch("scholar_rag.nodes.router.init_chat_model")
    def test_classifies_retrieval(self, mock_init):
        llm = MagicMock()
        response = MagicMock()
        response.content = "retrieval"
        llm.invoke.return_value = response
        mock_init.return_value = llm

        from scholar_rag.nodes.router import router_node
        result = router_node({"query": "Which papers cite this work?"})
        assert result["intent"] == "retrieval"

    @patch("scholar_rag.nodes.router.init_chat_model")
    def test_classifies_direct(self, mock_init):
        llm = MagicMock()
        response = MagicMock()
        response.content = "direct"
        llm.invoke.return_value = response
        mock_init.return_value = llm

        from scholar_rag.nodes.router import router_node
        result = router_node({"query": "What is a neural network?"})
        assert result["intent"] == "direct"

    @patch("scholar_rag.nodes.router.init_chat_model")
    def test_classifies_out_of_scope(self, mock_init):
        llm = MagicMock()
        response = MagicMock()
        response.content = "out_of_scope"
        llm.invoke.return_value = response
        mock_init.return_value = llm

        from scholar_rag.nodes.router import router_node
        result = router_node({"query": "Write me a Fibonacci program."})
        assert result["intent"] == "out_of_scope"

    @patch("scholar_rag.nodes.router.init_chat_model")
    def test_defaults_to_retrieval_on_unexpected_response(self, mock_init):
        llm = MagicMock()
        response = MagicMock()
        response.content = "I don't know"
        llm.invoke.return_value = response
        mock_init.return_value = llm

        from scholar_rag.nodes.router import router_node
        result = router_node({"query": "anything"})
        assert result["intent"] == "retrieval"


# ---------------------------------------------------------------------------
# Responder node
# ---------------------------------------------------------------------------

class TestResponderNode:
    @patch("scholar_rag.nodes.responder.init_chat_model")
    def test_synthesises_answer(self, mock_init):
        llm = MagicMock()
        response = MagicMock()
        response.content = "This paper [Author, 2020] demonstrated fuzzy classification."
        llm.invoke.return_value = response
        mock_init.return_value = llm

        from scholar_rag.nodes.responder import responder_node
        result = responder_node({
            "query": "What did the fuzzy paper do?",
            "retrieved_texts": ["Paper content here."],
            "answer": "",
        })
        assert "fuzzy classification" in result["answer"]
        assert "sources" in result

    @patch("scholar_rag.nodes.responder.init_chat_model")
    def test_handles_empty_context(self, mock_init):
        llm = MagicMock()
        response = MagicMock()
        response.content = "Insufficient context."
        llm.invoke.return_value = response
        mock_init.return_value = llm

        from scholar_rag.nodes.responder import responder_node
        result = responder_node({
            "query": "Something",
            "retrieved_texts": [],
            "answer": "",
        })
        assert result["answer"] == "Insufficient context."


# ---------------------------------------------------------------------------
# Full graph (all nodes mocked)
# ---------------------------------------------------------------------------

class TestBuildGraph:
    @patch("scholar_rag.nodes.router.init_chat_model")
    def test_out_of_scope_path_returns_refusal(self, mock_router_init):
        router_llm = MagicMock()
        router_response = MagicMock(); router_response.content = "out_of_scope"
        router_llm.invoke.return_value = router_response
        mock_router_init.return_value = router_llm

        graph = build_graph()
        result = graph.invoke({
            "query": "Write a Fibonacci program in Python.",
            "intent": "",
            "retrieved_texts": [],
            "agent_scratchpad": [],
            "answer": "",
            "sources": [],
        })

        assert result["intent"] == "out_of_scope"
        assert "scope" in result["answer"].lower()
        assert result["sources"] == []

    @patch("scholar_rag.nodes.responder.init_chat_model")
    @patch("scholar_rag.nodes.router.init_chat_model")
    def test_direct_path_skips_retriever(self, mock_router_init, mock_resp_init):
        router_llm = MagicMock()
        router_response = MagicMock(); router_response.content = "direct"
        router_llm.invoke.return_value = router_response
        mock_router_init.return_value = router_llm

        resp_llm = MagicMock()
        resp_response = MagicMock(); resp_response.content = "42 is the answer."
        resp_llm.invoke.return_value = resp_response
        mock_resp_init.return_value = resp_llm

        graph = build_graph()
        result = graph.invoke({
            "query": "What is 6 times 7?",
            "intent": "",
            "retrieved_texts": [],
            "agent_scratchpad": [],
            "answer": "",
            "sources": [],
        })

        assert result["intent"] == "direct"
        assert result["answer"] == "42 is the answer."

    @patch("scholar_rag.nodes.responder.init_chat_model")
    @patch("scholar_rag.nodes.retriever.create_agent")
    @patch("scholar_rag.nodes.router.init_chat_model")
    def test_retrieval_path_calls_retriever(
        self, mock_router_init, mock_create_agent, mock_resp_init
    ):
        # Router → retrieval
        router_llm = MagicMock()
        router_response = MagicMock(); router_response.content = "retrieval"
        router_llm.invoke.return_value = router_response
        mock_router_init.return_value = router_llm

        # ReAct agent
        from langchain_core.messages import AIMessage
        agent = MagicMock()
        ai_msg = AIMessage(content="Found relevant papers.")
        agent.invoke.return_value = {"messages": [ai_msg]}
        mock_create_agent.return_value = agent

        # Responder
        resp_llm = MagicMock()
        resp_response = MagicMock(); resp_response.content = "Synthesised answer."
        resp_llm.invoke.return_value = resp_response
        mock_resp_init.return_value = resp_llm

        graph = build_graph()
        result = graph.invoke({
            "query": "Which papers cite the fuzzy classifier?",
            "intent": "",
            "retrieved_texts": [],
            "agent_scratchpad": [],
            "answer": "",
            "sources": [],
        })

        assert result["intent"] == "retrieval"
        assert result["answer"] == "Synthesised answer."
        mock_create_agent.assert_called_once()
