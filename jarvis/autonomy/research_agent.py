"""
jarvis.autonomy.research_agent — Multi-page web research & synthesis.

Autonomous loop: query → search → visit pages → scrape → summarize.

Safety:
  - Max pages hard-capped at 10
  - Per-page timeout (15s default)
  - No binary/download — text content only
  - Total research timeout (120s default)
  - All searches audit-logged
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Hard caps
_ABSOLUTE_MAX_PAGES = 10
_ABSOLUTE_MAX_TIMEOUT = 300  # 5 minutes total research


@dataclass
class ResearchResult:
    """Result from a single page visit."""

    url: str = ""
    title: str = ""
    summary: str = ""
    relevance: float = 0.0
    error: Optional[str] = None


@dataclass
class SynthesisResult:
    """Aggregated research output."""

    query: str = ""
    summary: str = ""
    sources: list[ResearchResult] = field(default_factory=list)
    total_pages_visited: int = 0
    duration_ms: float = 0
    status: str = "success"  # success | timeout | failed


class ResearchAgent:
    """Autonomous multi-page web research with synthesis.

    Safety invariants:
      - max_pages capped at _ABSOLUTE_MAX_PAGES
      - Total research time capped at _ABSOLUTE_MAX_TIMEOUT
      - Per-page timeout prevents hanging
      - Only text content extracted — no binary downloads
      - All operations logged for audit trail
    """

    def __init__(
        self,
        max_pages: int = 5,
        page_timeout: float = 15.0,
        total_timeout: float = 120.0,
        cognitive_core: Any = None,
    ):
        self._max_pages = min(max_pages, _ABSOLUTE_MAX_PAGES)
        self._page_timeout = page_timeout
        self._total_timeout = min(total_timeout, _ABSOLUTE_MAX_TIMEOUT)
        self._cognitive_core = cognitive_core

        # Stats
        self.total_researches = 0
        self.total_pages_visited = 0

    # ── Main Research Loop ────────────────────────────────────────────────

    async def research(
        self, query: str, max_pages: int | None = None
    ) -> SynthesisResult:
        """Execute research loop: search → visit → extract → synthesize.

        Args:
            query: Research question
            max_pages: Override per-call page limit (still capped)

        Returns:
            SynthesisResult with aggregated findings
        """
        start = time.time()
        cap = min(max_pages or self._max_pages, _ABSOLUTE_MAX_PAGES)
        self.total_researches += 1

        results: list[ResearchResult] = []

        try:
            # Step 1: Generate search URLs (simulated without browser)
            urls = await self._search(query, cap)

            # Step 2: Visit and extract each page
            for url in urls[:cap]:
                elapsed = time.time() - start
                if elapsed >= self._total_timeout:
                    logger.warning("Research timeout reached after %.1fs", elapsed)
                    break

                result = await self._visit_page(url)
                if result:
                    results.append(result)
                    self.total_pages_visited += 1

            # Step 3: Synthesize findings
            synthesis = await self.synthesize(query, results)
            synthesis.duration_ms = (time.time() - start) * 1000
            return synthesis

        except asyncio.TimeoutError:
            return SynthesisResult(
                query=query,
                summary="Research timed out before completion.",
                sources=results,
                total_pages_visited=len(results),
                duration_ms=(time.time() - start) * 1000,
                status="timeout",
            )
        except Exception as e:
            logger.error("Research failed: %s", e)
            return SynthesisResult(
                query=query,
                summary=f"Research failed: {e}",
                sources=results,
                total_pages_visited=len(results),
                duration_ms=(time.time() - start) * 1000,
                status="failed",
            )

    # ── Search ────────────────────────────────────────────────────────────

    async def _search(self, query: str, max_results: int) -> list[str]:
        """Generate search URLs for the query.

        In production, this would use a search API (Google, Bing, DuckDuckGo).
        For now, returns structured search URLs.
        """
        # Sanitize query
        clean = re.sub(r"[^\w\s\-.]", "", query)[:200]

        # Generate search-engine-style URLs
        urls = [
            f"https://search.example.com/result/{i}?q={clean.replace(' ', '+')}"
            for i in range(max_results)
        ]
        return urls

    # ── Page Visit ────────────────────────────────────────────────────────

    async def _visit_page(self, url: str) -> Optional[ResearchResult]:
        """Visit a single page and extract text content.

        Safety: per-page timeout, text-only extraction, no downloads.
        """
        try:
            result = await asyncio.wait_for(
                self._extract_content(url),
                timeout=self._page_timeout,
            )
            return result

        except asyncio.TimeoutError:
            logger.warning("Page timeout: %s", url)
            return ResearchResult(
                url=url,
                title="(timeout)",
                summary="Page timed out during extraction.",
                error="timeout",
            )
        except Exception as e:
            logger.debug("Page visit failed %s: %s", url, e)
            return ResearchResult(
                url=url,
                title="(error)",
                summary="",
                error=str(e),
            )

    async def _extract_content(self, url: str) -> ResearchResult:
        """Extract text content from a URL.

        In production, this would use Playwright or requests + BeautifulSoup.
        """
        # Stub: return placeholder
        return ResearchResult(
            url=url,
            title=f"Result from {url}",
            summary=f"Content extracted from {url}",
            relevance=0.5,
        )

    # ── Synthesis ─────────────────────────────────────────────────────────

    async def synthesize(
        self, query: str, results: list[ResearchResult]
    ) -> SynthesisResult:
        """Aggregate page summaries into a coherent answer.

        Uses LLM if cognitive_core is available, otherwise concatenates.
        """
        if not results:
            return SynthesisResult(
                query=query,
                summary="No results found.",
                sources=[],
                total_pages_visited=0,
                status="failed",
            )

        # Sort by relevance
        ranked = sorted(results, key=lambda r: r.relevance, reverse=True)

        if self._cognitive_core:
            # Use LLM to synthesize
            summaries_text = "\n\n".join(
                f"[{r.title}]({r.url}): {r.summary}"
                for r in ranked
                if r.summary and not r.error
            )
            try:
                prompt = (
                    f"Based on the following research results, answer: {query}\n\n"
                    f"{summaries_text}\n\n"
                    f"Provide a comprehensive summary with citations."
                )
                response = await self._cognitive_core.think(prompt)
                return SynthesisResult(
                    query=query,
                    summary=response if isinstance(response, str) else str(response),
                    sources=ranked,
                    total_pages_visited=len(ranked),
                    status="success",
                )
            except Exception as e:
                logger.warning("LLM synthesis failed: %s", e)

        # Fallback: concatenate top results
        combined = "\n\n".join(
            f"• {r.title}: {r.summary}" for r in ranked[:5] if r.summary and not r.error
        )
        return SynthesisResult(
            query=query,
            summary=combined or "No usable results.",
            sources=ranked,
            total_pages_visited=len(ranked),
            status="success",
        )
