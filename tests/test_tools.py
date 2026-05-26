"""Tests for the agent tools (Semantic Scholar + Qdrant search)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Semantic Scholar tools
# ---------------------------------------------------------------------------

class TestGetPaperDetails:
    @patch("scholar_rag.tools.semantic_scholar.SemanticScholar")
    def test_returns_formatted_string(self, mock_s2_cls):
        paper = MagicMock()
        paper.paperId = "abc123"
        paper.title = "A Great Paper"
        paper.year = 2020
        paper.venue = "NeurIPS"
        paper.citationCount = 42
        paper.abstract = "This paper does stuff."
        paper.tldr = {"text": "Short summary."}
        a1 = MagicMock(); a1.name = "Alice"
        paper.authors = [a1]

        mock_s2_cls.return_value.get_paper.return_value = paper

        from scholar_rag.tools.semantic_scholar import get_paper_details
        result = get_paper_details.invoke({"paper_id": "abc123"})

        assert "A Great Paper" in result
        assert "2020" in result
        assert "NeurIPS" in result
        assert "42" in result
        assert "Short summary." in result

    @patch("scholar_rag.tools.semantic_scholar.SemanticScholar")
    def test_handles_api_error(self, mock_s2_cls):
        mock_s2_cls.return_value.get_paper.side_effect = Exception("timeout")
        from scholar_rag.tools.semantic_scholar import get_paper_details
        result = get_paper_details.invoke({"paper_id": "bad_id"})
        assert "Error" in result


class TestGetCitations:
    @patch("scholar_rag.tools.semantic_scholar.SemanticScholar")
    def test_returns_citing_papers(self, mock_s2_cls):
        paper = MagicMock()
        paper.title = "Seed Paper"

        c1 = MagicMock()
        c1.paperId = "cit1"; c1.title = "Citing Paper"; c1.year = 2022
        c1.citationCount = 10
        a = MagicMock(); a.name = "Bob"
        c1.authors = [a]

        paper.citations = [c1]
        mock_s2_cls.return_value.get_paper.return_value = paper

        from scholar_rag.tools.semantic_scholar import get_citations
        result = get_citations.invoke({"paper_id": "seed1", "limit": 5})

        assert "Citing Paper" in result
        assert "cit1" in result

    @patch("scholar_rag.tools.semantic_scholar.SemanticScholar")
    def test_empty_citations(self, mock_s2_cls):
        paper = MagicMock()
        paper.title = "Paper"
        paper.citations = []
        mock_s2_cls.return_value.get_paper.return_value = paper

        from scholar_rag.tools.semantic_scholar import get_citations
        result = get_citations.invoke({"paper_id": "x"})
        assert "No citations" in result


class TestGetReferences:
    @patch("scholar_rag.tools.semantic_scholar.SemanticScholar")
    def test_returns_references(self, mock_s2_cls):
        paper = MagicMock()
        paper.title = "Main Paper"

        r1 = MagicMock()
        r1.paperId = "ref1"; r1.title = "Foundation Paper"; r1.year = 2010
        r1.citationCount = 500
        a = MagicMock(); a.name = "Carol"
        r1.authors = [a]

        paper.references = [r1]
        mock_s2_cls.return_value.get_paper.return_value = paper

        from scholar_rag.tools.semantic_scholar import get_references
        result = get_references.invoke({"paper_id": "main1", "limit": 5})

        assert "Foundation Paper" in result
        assert "ref1" in result


# ---------------------------------------------------------------------------
# Qdrant hybrid search
# ---------------------------------------------------------------------------

class TestSearchCorpus:
    @patch("scholar_rag.tools.qdrant_search._get_client")
    @patch("scholar_rag.tools.qdrant_search._get_models")
    def test_returns_formatted_results(self, mock_get_models, mock_get_client):
        # Mock embedding models
        dense_model = MagicMock()
        dense_model.encode.return_value = np.zeros(384)

        sparse_emb = MagicMock()
        sparse_emb.indices = np.array([0, 1])
        sparse_emb.values = np.array([0.5, 0.3])
        sparse_model = MagicMock()
        sparse_model.embed.return_value = iter([sparse_emb])
        mock_get_models.return_value = (dense_model, sparse_model)

        # Mock Qdrant result
        point = MagicMock()
        point.score = 0.95
        point.payload = {
            "paper_id": "p1",
            "title": "Fuzzy Systems Paper",
            "authors": ["Oliver Buchtala"],
            "year": 2005,
            "venue": "IEEE SMC",
            "citation_count": 20,
            "source": "seed",
            "text": "This paper presents a fuzzy approach to classification.",
        }
        result_mock = MagicMock()
        result_mock.points = [point]
        mock_get_client.return_value.query_points.return_value = result_mock

        from scholar_rag.tools.qdrant_search import search_corpus
        result = search_corpus.invoke({"query": "fuzzy classification", "top_k": 5})

        assert "Fuzzy Systems Paper" in result
        assert "2005" in result
        assert "IEEE SMC" in result

    @patch("scholar_rag.tools.qdrant_search._get_client")
    @patch("scholar_rag.tools.qdrant_search._get_models")
    def test_no_results_message(self, mock_get_models, mock_get_client):
        dense_model = MagicMock()
        dense_model.encode.return_value = np.zeros(384)
        sparse_emb = MagicMock()
        sparse_emb.indices = np.array([0])
        sparse_emb.values = np.array([1.0])
        sparse_model = MagicMock()
        sparse_model.embed.return_value = iter([sparse_emb])
        mock_get_models.return_value = (dense_model, sparse_model)

        result_mock = MagicMock()
        result_mock.points = []
        mock_get_client.return_value.query_points.return_value = result_mock

        from scholar_rag.tools.qdrant_search import search_corpus
        result = search_corpus.invoke({"query": "unknown topic"})
        assert "No relevant papers" in result
