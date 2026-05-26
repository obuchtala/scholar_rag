"""Tests for the ingestion pipeline."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from scholar_rag.ingest import (
    _paper_to_dict,
    build_document_text,
    ensure_collection,
    expand_corpus,
    fetch_seed_papers,
    ingest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_s2_paper(
    paper_id="abc123",
    title="Test Paper",
    abstract="A test abstract.",
    year=2020,
    venue="IEEE SMC",
    authors=None,
    citation_count=5,
    tldr=None,
):
    """Build a mock Semantic Scholar paper object."""
    p = MagicMock()
    p.paperId = paper_id
    p.title = title
    p.abstract = abstract
    p.year = year
    p.venue = venue
    p.citationCount = citation_count
    p.tldr = tldr

    if authors is None:
        authors = ["Alice", "Bob"]
    author_mocks = []
    for name in authors:
        a = MagicMock()
        a.name = name
        author_mocks.append(a)
    p.authors = author_mocks

    return p


# ---------------------------------------------------------------------------
# build_document_text
# ---------------------------------------------------------------------------

class TestBuildDocumentText:
    def test_title_abstract_tldr(self):
        paper = {
            "title": "My Paper",
            "abstract": "Some abstract.",
            "tldr": "A short summary.",
        }
        text = build_document_text(paper)
        assert "My Paper" in text
        assert "Some abstract." in text
        assert "TLDR: A short summary." in text

    def test_missing_abstract(self):
        paper = {"title": "No Abstract", "abstract": "", "tldr": None}
        text = build_document_text(paper)
        assert text == "No Abstract"

    def test_missing_tldr(self):
        paper = {"title": "T", "abstract": "A", "tldr": None}
        text = build_document_text(paper)
        assert "TLDR" not in text

    def test_empty_paper(self):
        assert build_document_text({}) == ""


# ---------------------------------------------------------------------------
# _paper_to_dict
# ---------------------------------------------------------------------------

class TestPaperToDict:
    def test_basic_fields(self):
        p = _make_s2_paper(tldr=None)
        d = _paper_to_dict(p, source="seed")
        assert d["paper_id"] == "abc123"
        assert d["title"] == "Test Paper"
        assert d["year"] == 2020
        assert d["source"] == "seed"
        assert d["authors"] == ["Alice", "Bob"]

    def test_tldr_dict(self):
        p = _make_s2_paper(tldr={"text": "Short summary.", "model": "x"})
        d = _paper_to_dict(p, source="seed")
        assert d["tldr"] == "Short summary."

    def test_tldr_object(self):
        tldr = MagicMock()
        tldr.text = "Object summary."
        p = _make_s2_paper(tldr=tldr)
        d = _paper_to_dict(p, source="expanded")
        assert d["tldr"] == "Object summary."

    def test_no_references_field(self):
        d = _paper_to_dict(_make_s2_paper(), source="seed")
        assert "references" not in d


# ---------------------------------------------------------------------------
# fetch_seed_papers
# ---------------------------------------------------------------------------

class TestFetchSeedPapers:
    def test_returns_papers(self):
        sch = MagicMock()

        author_result = MagicMock()
        author_result.authorId = "author1"
        author_result.name = "Oliver Buchtala"
        sch.search_author.return_value = [author_result]

        author_detail = MagicMock()
        p1 = MagicMock(); p1.paperId = "p1"; p1.title = "Paper One"; p1.year = 2005
        p2 = MagicMock(); p2.paperId = "p2"; p2.title = "Paper Two"; p2.year = 2008
        author_detail.papers = [p1, p2]
        sch.get_author.return_value = author_detail

        sch.get_paper.side_effect = [
            _make_s2_paper(paper_id="p1", title="Paper One"),
            _make_s2_paper(paper_id="p2", title="Paper Two"),
        ]

        papers = fetch_seed_papers("Oliver Buchtala", sch)
        assert len(papers) == 2
        assert papers[0]["paper_id"] == "p1"

    def test_raises_when_author_not_found(self):
        sch = MagicMock()
        sch.search_author.return_value = []
        with pytest.raises(ValueError, match="No author found"):
            fetch_seed_papers("Nobody", sch)

    def test_skips_failed_paper_fetch(self):
        sch = MagicMock()
        author_result = MagicMock()
        author_result.authorId = "a1"; author_result.name = "Test Author"
        sch.search_author.return_value = [author_result]

        author_detail = MagicMock()
        p1 = MagicMock(); p1.paperId = "p1"; p1.title = "P1"; p1.year = 2020
        author_detail.papers = [p1]
        sch.get_author.return_value = author_detail

        sch.get_paper.side_effect = Exception("API error")
        papers = fetch_seed_papers("Test Author", sch)
        assert papers == []


# ---------------------------------------------------------------------------
# expand_corpus
# ---------------------------------------------------------------------------

class TestExpandCorpus:
    @patch("scholar_rag.ingest._get_citing_paper_ids")
    def test_fetches_citing_papers(self, mock_citing):
        seed = [{"paper_id": "seed1"}]
        mock_citing.return_value = ["cit1", "cit2"]
        sch = MagicMock()
        sch.get_paper.side_effect = [
            _make_s2_paper(paper_id="cit1", title="Citing One"),
            _make_s2_paper(paper_id="cit2", title="Citing Two"),
        ]
        expanded = expand_corpus(seed, sch)
        assert len(expanded) == 2

    @patch("scholar_rag.ingest._get_citing_paper_ids")
    def test_does_not_re_fetch_seeds(self, mock_citing):
        seed = [{"paper_id": "seed1"}]
        mock_citing.return_value = ["seed1", "cit1"]
        sch = MagicMock()
        sch.get_paper.return_value = _make_s2_paper(paper_id="cit1", title="Citing One")
        expand_corpus(seed, sch)
        for c in sch.get_paper.call_args_list:
            assert "seed1" not in c.args

    @patch("scholar_rag.ingest._get_citing_paper_ids")
    def test_skips_papers_without_abstract(self, mock_citing):
        seed = [{"paper_id": "s1"}]
        mock_citing.return_value = ["cit1"]
        sch = MagicMock()
        sch.get_paper.return_value = _make_s2_paper(paper_id="cit1", title="Title", abstract="")
        expanded = expand_corpus(seed, sch)
        assert expanded == []


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------

class TestEnsureCollection:
    def test_creates_collection_if_missing(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        ensure_collection(client, "papers")
        client.create_collection.assert_called_once()
        call_kwargs = client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "papers"
        assert "dense" in call_kwargs["vectors_config"]
        assert "sparse" in call_kwargs["sparse_vectors_config"]

    def test_skips_if_already_exists(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        ensure_collection(client, "papers")
        client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# ingest (integration-level, all external deps mocked)
# ---------------------------------------------------------------------------

class TestIngest:
    @patch("scholar_rag.ingest.SemanticScholar")
    @patch("scholar_rag.ingest.QdrantClient")
    @patch("scholar_rag.ingest.SentenceTransformer")
    @patch("scholar_rag.ingest.SparseTextEmbedding")
    def test_upserts_papers(
        self, mock_sparse_cls, mock_dense_cls, mock_qdrant_cls, mock_s2_cls
    ):
        # Semantic Scholar
        sch = MagicMock()
        mock_s2_cls.return_value = sch

        author_result = MagicMock()
        author_result.authorId = "a1"; author_result.name = "Oliver Buchtala"
        sch.search_author.return_value = [author_result]

        author_detail = MagicMock()
        p_ref = MagicMock(); p_ref.paperId = "p1"; p_ref.title = "P1"; p_ref.year = 2020
        author_detail.papers = [p_ref]
        sch.get_author.return_value = author_detail
        sch.get_paper.return_value = _make_s2_paper(paper_id="p1", title="Paper One")

        # Dense embeddings — return a (1, 384) float32 array
        dense_model = MagicMock()
        dense_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_dense_cls.return_value = dense_model

        # Sparse embeddings
        sparse_emb = MagicMock()
        sparse_emb.indices = np.array([0, 1])
        sparse_emb.values = np.array([0.5, 0.3])
        sparse_model = MagicMock()
        sparse_model.embed.return_value = iter([sparse_emb])
        mock_sparse_cls.return_value = sparse_model

        # Qdrant
        qdrant = MagicMock()
        qdrant.collection_exists.return_value = False
        mock_qdrant_cls.return_value = qdrant

        count = ingest(
            author="Oliver Buchtala",
            expand_hops=0,
            qdrant_url="http://localhost:6333",
            collection="papers",
        )

        assert count == 1
        qdrant.upsert.assert_called_once()
        upsert_kwargs = qdrant.upsert.call_args.kwargs
        assert upsert_kwargs["collection_name"] == "papers"
        assert len(upsert_kwargs["points"]) == 1
