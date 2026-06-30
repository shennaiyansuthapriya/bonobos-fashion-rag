"""
System and user prompt templates for Bonobos fashion product discovery RAG.
"""

SYSTEM_PROMPT = """You are a personal style advisor for Bonobos, the premium men's fashion brand known for \
exceptional fit, quality fabrics, and modern design. Bonobos is owned by Walmart and specialises in chinos, \
dress pants, suits, shirts, and casual wear with a focus on fit innovation.

Your role is to help customers discover and style products that match their lifestyle, occasion, and aesthetic.

Guidelines:
- Recommend specific products with name, category, fit type, colour, material, and price
- Suggest complete outfit combinations when appropriate (e.g., pants + shirt + shoes)
- Reference occasion (business casual, smart casual, wedding, weekend) to narrow recommendations
- Mention Bonobos' signature fits: Slim, Athletic, Straight, Tailored, Relaxed
- Explain how to style items and what they pair well with
- Include care and fabric details when material quality is relevant
- Format product citations as [Product Name] with fit and colour
- Use conversational, aspirational tone — avoid overly technical fashion jargon
"""

USER_PROMPT_TEMPLATE = """Based on the following product and style guide information, provide personalised \
fashion recommendations.

PRODUCT & STYLE CONTEXT:
{context}

CUSTOMER QUESTION:
{query}

{filter_context}

Provide specific product recommendations with styling advice. Where applicable, suggest complete outfit combinations \
and explain how the pieces work together. Reference Bonobos fit options to help the customer choose the right cut."""


def build_context(results: list[dict]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        text = r.get("text", "")
        source = r.get("source", "product")
        name = meta.get("name", "Unknown")
        category = meta.get("category", "")
        fit = meta.get("fit_type", "")
        color = meta.get("color", "")
        color_family = meta.get("color_family", "")
        material = meta.get("material", "")
        price = meta.get("price_usd")
        sale_price = meta.get("sale_price_usd")
        occasions = meta.get("occasion", [])
        pattern = meta.get("pattern", "")

        if source == "product":
            price_str = f"${price:.0f}" if price else ""
            if sale_price and sale_price < (price or float("inf")):
                price_str = f"~~${price:.0f}~~ ${sale_price:.0f}"
            occasion_str = ", ".join(occasions) if occasions else ""
            spec_parts = [s for s in [fit, color, pattern, material, price_str, occasion_str] if s]
            lines.append(
                f"[{i}] {name} ({category})\n"
                f"    {' | '.join(spec_parts)}\n"
                f"    {text}"
            )
        else:
            lines.append(f"[{i}] STYLE GUIDE\n    {text}")

    return "\n\n".join(lines)


def build_filter_context(category: str | None, occasion: str | None) -> str:
    parts = []
    if category:
        parts.append(f"Category: {category}")
    if occasion:
        parts.append(f"Occasion: {occasion}")
    return ("Style filters: " + ", ".join(parts)) if parts else ""
