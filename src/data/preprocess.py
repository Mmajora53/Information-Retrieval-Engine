"""Preprocessing helpers used to build the shared `content` field."""

import re


def to_string(v):
    """Return a safe string representation of a value."""
    if v is None:
        return ""
    return str(v).replace("\ufffd", " ").strip()


def build_content(item: dict) -> str:
    """
    Build the merged text representation used by retrieval and classification.

    Parameters
    ----------
    item : dict
        Record that may contain `title`, `text`, and `tags`.

    Returns
    -------
    str
        Concatenated text content.
    """
    parts = []

    title = to_string(item.get("title"))
    if title:
        parts.append(title)

    text = to_string(item.get("text"))
    if text:
        parts.append(text)

    tags = item.get("tags")

    if isinstance(tags, list):
        valid_tags = [to_string(t) for t in tags if to_string(t)]
        if valid_tags:
            parts.append(" ".join(valid_tags))
    else:
        tag_str = to_string(tags)
        if tag_str:
            parts.append(tag_str)

    return " ".join(parts)


def clean_text(text: str) -> str:
    """
    Apply light normalization to a text string.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    return text.strip()


def process_items(items: list[dict], clean: bool) -> list[dict]:
    """Copy items, add `content`, and optionally normalize it."""
    enriched_items = []
    for item in items:
        new_item = item.copy()
        content = build_content(new_item)
        if clean:
            content = clean_text(content)

        new_item["content"] = content
        enriched_items.append(new_item)
    return enriched_items


def add_content_field(docs: list[dict], queries: list[dict], clean: bool) -> tuple[list[dict], list[dict]]:
    """
    Add a `content` field to documents and queries.

    Parameters
    ----------
    docs : list[dict]
        Document records.
    queries : list[dict]
        Query records.
    clean : bool
        Whether to apply `clean_text` to the generated content.

    Returns
    -------
    tuple[list[dict], list[dict]]
        Enriched documents and queries.
    """
    docs_enriched = process_items(docs, clean)
    queries_enriched = process_items(queries, clean)

    return docs_enriched, queries_enriched
