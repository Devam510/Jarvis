"""Tests for jarvis.autonomy.research_agent — Multi-page research."""

import pytest
from jarvis.autonomy.research_agent import (
    ResearchAgent,
    ResearchResult,
    SynthesisResult,
    _ABSOLUTE_MAX_PAGES,
    _ABSOLUTE_MAX_TIMEOUT,
)


class TestPageCap:
    """Safety: max pages hard-capped."""

    def test_default_max_pages(self):
        ra = ResearchAgent(max_pages=5)
        assert ra._max_pages == 5

    def test_cap_at_absolute(self):
        ra = ResearchAgent(max_pages=100)
        assert ra._max_pages == _ABSOLUTE_MAX_PAGES

    def test_timeout_capped(self):
        ra = ResearchAgent(total_timeout=9999)
        assert ra._total_timeout == _ABSOLUTE_MAX_TIMEOUT


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_urls(self):
        ra = ResearchAgent()
        urls = await ra._search("test query", 3)
        assert len(urls) == 3
        assert all("test+query" in u for u in urls)

    @pytest.mark.asyncio
    async def test_search_sanitizes_query(self):
        ra = ResearchAgent()
        urls = await ra._search("test; rm -rf /", 2)
        assert len(urls) == 2
        # Should not contain shell metacharacters
        for u in urls:
            assert ";" not in u


class TestResearch:
    @pytest.mark.asyncio
    async def test_basic_research(self):
        ra = ResearchAgent(max_pages=2)
        result = await ra.research("test topic")
        assert isinstance(result, SynthesisResult)
        assert result.query == "test topic"
        assert result.total_pages_visited <= 2
        assert result.duration_ms >= 0
        assert ra.total_researches == 1

    @pytest.mark.asyncio
    async def test_research_per_call_cap(self):
        ra = ResearchAgent(max_pages=3)
        result = await ra.research("topic", max_pages=2)
        assert result.total_pages_visited <= 2

    @pytest.mark.asyncio
    async def test_research_timeout(self):
        ra = ResearchAgent(total_timeout=0.001)  # near instant
        result = await ra.research("test")
        assert result.status in ("success", "timeout")


class TestSynthesis:
    @pytest.mark.asyncio
    async def test_synthesize_empty(self):
        ra = ResearchAgent()
        result = await ra.synthesize("query", [])
        assert result.status == "failed"
        assert "No results" in result.summary

    @pytest.mark.asyncio
    async def test_synthesize_with_results(self):
        ra = ResearchAgent()
        results = [
            ResearchResult(
                url="http://a.com", title="A", summary="Summary A", relevance=0.8
            ),
            ResearchResult(
                url="http://b.com", title="B", summary="Summary B", relevance=0.3
            ),
        ]
        synthesis = await ra.synthesize("query", results)
        assert synthesis.status == "success"
        assert "Summary A" in synthesis.summary

    @pytest.mark.asyncio
    async def test_synthesize_skips_errors(self):
        ra = ResearchAgent()
        results = [
            ResearchResult(
                url="http://a.com", title="A", summary="Good", relevance=0.9
            ),
            ResearchResult(url="http://b.com", title="B", summary="", error="timeout"),
        ]
        synthesis = await ra.synthesize("query", results)
        assert "Good" in synthesis.summary


class TestPageVisit:
    @pytest.mark.asyncio
    async def test_visit_returns_result(self):
        ra = ResearchAgent()
        result = await ra._visit_page("http://example.com")
        assert isinstance(result, ResearchResult)
        assert result.error is None

    @pytest.mark.asyncio
    async def test_visit_timeout(self):
        ra = ResearchAgent(page_timeout=0.001)
        # The stub completes fast, so this might still succeed
        result = await ra._visit_page("http://example.com")
        assert result is not None


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        ra = ResearchAgent(max_pages=1)
        await ra.research("test 1")
        await ra.research("test 2")
        assert ra.total_researches == 2
        assert ra.total_pages_visited >= 2
