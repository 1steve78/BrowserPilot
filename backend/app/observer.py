"""Page observer module using Playwright.

Responsible for inspecting the DOM, identifying interactive elements,
and distilling the page state into a structured PageObservation.

Owned by: Browser Automation Engineer
"""

from typing import List, Optional
from playwright.async_api import Page
from .schemas import ElementDescriptor, PageObservation


class PlaywrightObserver:
    """Extracts structured observations from an active Playwright page."""

    def __init__(self, max_snippet_length: int = 1500) -> None:
        self.max_snippet_length = max_snippet_length

    async def observe(self, page: Page) -> PageObservation:
        """Capture the current state of the page.

        Extracts title, URL, interactive elements, text snippet, and DOM summary.
        """
        url = page.url
        title = await page.title()

        # Extract interactive elements (buttons, inputs, links, selects)
        interactive_elements = await self._extract_interactive_elements(page)

        # Build token-efficient DOM summary for LLM
        dom_summary = self._build_dom_summary(interactive_elements)

        # Extract visible text snippet
        snippet = await self._extract_text_snippet(page)

        return PageObservation(
            url=url,
            title=title,
            dom_summary=dom_summary,
            interactive_elements=interactive_elements,
            page_text_snippet=snippet,
        )

    async def _extract_interactive_elements(self, page: Page) -> List[ElementDescriptor]:
        """Find clickable, typeable, and navigable elements on the active page."""
        elements: List[ElementDescriptor] = []

        # Starter implementation: query common interactive tags
        selector_query = "button, a[href], input, select, textarea, [role='button']"
        try:
            locators = page.locator(selector_query)
            count = await locators.count()
            for i in range(min(count, 30)):  # Cap to prevent context explosion
                loc = locators.nth(i)
                if not await loc.is_visible():
                    continue

                tag = await loc.evaluate("el => el.tagName.toLowerCase()")
                text = (await loc.text_content() or "").strip()
                aria = await loc.get_attribute("aria-label") or ""
                dom_id = await loc.get_attribute("id") or None

                computed_selector = f"#{dom_id}" if dom_id else f"{tag}:nth-of-type({i+1})"

                elements.append(
                    ElementDescriptor(
                        id=dom_id,
                        tag_name=tag,
                        selector=computed_selector,
                        text=text[:80] if text else None,
                        aria_label=aria if aria else None,
                        is_interactive=True,
                    )
                )
        except Exception:
            # Fallback for transient DOM detachment
            pass

        # TODO (Browser Automation Engineer): Implement robust accessibility-tree walking
        # TODO (Browser Automation Engineer): Add bounding box coordinate extraction
        return elements

    def _build_dom_summary(self, elements: List[ElementDescriptor]) -> str:
        """Format interactive elements into a concise markdown list for the model."""
        lines = []
        for i, el in enumerate(elements):
            label = el.text or el.aria_label or el.id or "unlabeled"
            lines.append(f"[{i}] <{el.tag_name}> '{label}' -> selector: `{el.selector}`")
        return "\n".join(lines)

    async def _extract_text_snippet(self, page: Page) -> str:
        """Extract main body text truncated for token limits."""
        try:
            body_text = await page.locator("body").inner_text()
            return body_text[: self.max_snippet_length].strip()
        except Exception:
            return ""

        # TODO (Browser Automation Engineer): Filter out scripts, styles, and cookie banners
