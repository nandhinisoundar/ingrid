"""Barcode -> Pinecone product -> OpenAI ingredient health assessment."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from barcode_detector import extract_text

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
PRODUCT_INDEX_NAME = os.getenv("PINECONE_PRODUCT_INDEX", "ingrid-beverages")
VALID_VERDICTS = {"ok", "care", "avoid", "unknown"}
BANNED_PHRASES = [
    "cures", "will prevent", "treats your", "safe for you",
    "you should stop eating", "diagnos", "prescrib",
]


def get_openai_client():
    """Return an OpenAI client only when the API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return OpenAI(api_key=api_key) if api_key else None


def _extract_nutrition(product_text):
    """Read the per-100g nutrition JSON preserved in the indexed product text."""
    match = re.search(
        r"Nutrition per 100g: (\[.*?\])(?:\nNutri-Score:|$)",
        product_text,
        flags=re.DOTALL,
    )
    if not match:
        return {}

    try:
        rows = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}

    return {
        row["name"]: {"value": row["100g"], "unit": row.get("unit", "")}
        for row in rows
        if isinstance(row, dict) and row.get("name") and row.get("100g") is not None
    }


def _extract_additives_and_highlights(metadata):
    """Derive additive codes and notable ingredient tags from product metadata."""
    tags = [tag.strip() for tag in str(metadata.get("ingredients_tags", "")).split(";") if tag.strip()]
    additives = []
    highlights = []
    for tag in tags:
        label = tag.removeprefix("en:").replace("-", " ").title()
        if re.fullmatch(r"en:e\d+[a-z]*", tag.lower()):
            additives.append(label)
        elif tag.startswith("en:"):
            highlights.append(label)
    return {
        "additives": list(dict.fromkeys(additives)),
        "highlights": list(dict.fromkeys(highlights))[:12],
    }


def lookup_pinecone_barcode(barcode):
    """Fetch a product from Pinecone using its exact barcode vector ID."""
    code = str(barcode).strip()
    if not re.fullmatch(r"\d{8,14}", code):
        return None

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        index = Pinecone(api_key=api_key).Index(PRODUCT_INDEX_NAME)
        vector = index.fetch(ids=[code]).vectors.get(code)
    except Exception:
        return None

    if not vector:
        return None

    metadata = vector.metadata or {}
    return {
        "details": metadata,
        "ingredients_text": metadata.get("ingredients_text", ""),
        "nutrition": _extract_nutrition(metadata.get("text", "")),
        "source": f"Pinecone: {PRODUCT_INDEX_NAME}",
    }


def read_barcode(source):
    """Return a barcode from a raw code or barcode-image path."""
    if source is None:
        return ""
    candidate = str(source).strip()
    if not candidate:
        return ""
    try:
        if Path(candidate).is_file():
            barcode, _ = extract_text(candidate)
            return barcode
    except OSError:
        pass
    return candidate


def parse_ingredients(ingredients_text):
    """Split product ingredient text into clean, unique labels."""
    cleaned = str(ingredients_text or "").lower()
    if "ingredients" in cleaned:
        cleaned = cleaned.split("ingredients", 1)[1].lstrip(": ")
    cleaned = cleaned.replace("(", ",").replace(")", ",")
    cleaned = re.sub(r"[;/\n]+", ",", cleaned)
    ingredients = []
    for part in cleaned.split(","):
        ingredient = re.sub(r"\s+", " ", part).strip(" .,:()")
        if ingredient and len(ingredient) > 1:
            ingredients.append(ingredient)
    return list(dict.fromkeys(ingredients))


def _health_prompt(product, ingredients, nutriscore_grade):
    return f"""You are a cautious food product analyst. Assess the product using ONLY the supplied product details, ingredient list, nutrition values, and Nutri-Score.

Return valid JSON with exactly these keys:
- verdicts: object mapping every listed ingredient to ok, care, avoid, or unknown
- summary: short plain-language product assessment
- ingredient_notes: object mapping each ingredient to one short reason

Rules:
- Nutri-Score is a product-level signal, not proof an ingredient is healthy or harmful.
- Use care for high sugar, salt, saturated fat, or ingredients that deserve moderation.
- Use avoid only for a clearly established serious concern; use unknown when information is insufficient.
- Do not provide medical, dietary, or treatment advice.
- Do not say a product is safe or unsafe for a person or condition.
- Every verdict key must exactly match an item in the ingredient list.

PRODUCT DETAILS
{json.dumps(product, ensure_ascii=False)}

INGREDIENTS
{json.dumps(ingredients, ensure_ascii=False)}

NUTRI-SCORE
{nutriscore_grade or "Not available"}
"""


def check_output(text):
    """Flag medical-claim language in generated output."""
    lowered = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def analyse_label(source, skip_llm=False):
    """Fetch barcode product details and generate LLM ingredient verdicts."""
    barcode = read_barcode(source)
    product_record = lookup_pinecone_barcode(barcode)
    if not product_record:
        return {
            "ingredients": [], "verdicts": {}, "matches": {}, "unknowns": [],
            "product": None, "nutriscore_grade": None, "nutrition": {},
            "additives_highlights": {"additives": [], "highlights": []},
            "explanation": f"Barcode {barcode or 'not detected'} was not found in the Pinecone product index.",
            "violations": [], "external_source": None,
        }

    product = product_record["details"]
    ingredients = parse_ingredients(product_record["ingredients_text"])
    result = {
        "ingredients": ingredients,
        "verdicts": {ingredient: "unknown" for ingredient in ingredients},
        "matches": {ingredient: None for ingredient in ingredients},
        "unknowns": list(ingredients),
        "product": product,
        "nutriscore_grade": product.get("nutriscore_grade"),
        "nutrition": product_record["nutrition"],
        "additives_highlights": _extract_additives_and_highlights(product),
        "explanation": None,
        "violations": [],
        "external_source": product_record["source"],
    }

    client = get_openai_client()
    if skip_llm or client is None:
        result["explanation"] = "LLM health assessment disabled. Product details were retrieved from Pinecone."
        return result

    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": _health_prompt(product, ingredients, result["nutriscore_grade"])}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        assessment = json.loads(response.choices[0].message.content)
        model_verdicts = assessment.get("verdicts", {})
        result["verdicts"] = {
            ingredient: model_verdicts.get(ingredient, "unknown")
            if model_verdicts.get(ingredient, "unknown") in VALID_VERDICTS
            else "unknown"
            for ingredient in ingredients
        }
        result["unknowns"] = [ingredient for ingredient, verdict in result["verdicts"].items() if verdict == "unknown"]
        notes = assessment.get("ingredient_notes", {})
        summary = assessment.get("summary", "No health assessment was returned.")
        if isinstance(notes, dict) and notes:
            summary += "\n\n" + "\n".join(f"{ingredient}: {notes.get(ingredient, 'No explanation returned.')}" for ingredient in ingredients)
        violations = check_output(summary)
        result["explanation"] = "Assessment withheld because it contained medical-claim language." if violations else summary
        result["violations"] = violations
    except Exception as exc:
        result["explanation"] = f"(Health assessment unavailable: {exc})"

    return result
