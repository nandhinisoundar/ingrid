# Ingredient Health Checker — Complete Build Guide

A RAG application that reads a food label, matches each ingredient against a
curated regulatory knowledge base, and returns a rating with citations.

**Stack:** Python · LangChain · Chroma · EasyOCR · Streamlit · Claude API

**Realistic time budget**

| Phase | Time |
|---|---|
| Setup and scaffolding | 30 min |
| Knowledge base (the real work) | 3–4 hours |
| Code (all five files) | 2 hours |
| Evaluation and ablations | 2 hours |
| Report | 3 hours |

---

## Step 0 — Prerequisites

You need Python 3.10 or newer and an Anthropic API key from
`console.anthropic.com`.

```bash
python --version    # must be 3.10+
```

Create the project and a virtual environment:

```bash
mkdir ingredient-checker && cd ingredient-checker
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## Step 1 — Install dependencies

Create `requirements.txt`:

```
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-chroma>=0.2.0
langchain-huggingface>=0.1.0
langchain-community>=0.3.0
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
chromadb>=0.5.0
easyocr>=1.7.0
pillow>=10.0.0
streamlit>=1.38.0
python-dotenv>=1.0.0
pandas>=2.0.0
```

```bash
pip install -r requirements.txt
```

> This is a large install — EasyOCR pulls in PyTorch, roughly 2 GB. Start it
> before you do anything else and let it run in the background.
>
> LangChain's package layout changes fairly often. If an import fails, check the
> current docs for the correct module path rather than assuming the code below is
> wrong. Pin whatever versions work for you and record them in your report so the
> result is reproducible.

Create `.env` in the project root:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Create `.gitignore`:

```
venv/
.env
chroma_db/
__pycache__/
*.pyc
```

---

## Step 2 — Project structure

Create these files now, empty. You will fill them in order.

```
ingredient-checker/
├── .env                  # API key, never commit
├── .gitignore
├── requirements.txt
├── kb.json               # knowledge base — your main deliverable
├── build_index.py        # run once, builds the vector store
├── barcode_detector.py   # image → barcode
├── chain.py              # the RAG pipeline
├── app.py                # Streamlit UI
├── evaluate.py           # metrics + ablations
├── test_labels.json      # gold set for evaluation
├── chroma_db/            # generated, do not edit
└── images/               # test photos for the OCR eval
```

---

## Step 3 — Build the knowledge base

**This is the part that determines your grade.** The code is mechanical; the
knowledge base is where the intellectual work is.

### 3.1 Write down your rubric first

Before rating anything, commit to a rule and state it in your report. Suggested:

```
avoid  → withdrawn, banned, or restricted in a major jurisdiction,
         OR classified by IARC in Group 1 / 2A / 2B,
         OR industrial trans fat

care   → permitted, but carries a numerical Acceptable Daily Intake,
         OR carries a mandatory warning label,
         OR regulators have flagged data gaps

ok     → regulator judged that no numerical ADI was necessary,
         OR it is an essential nutrient or a basic foodstuff
```

Having a written rubric means your ratings are reproducible by someone else.
Without one, they are opinions.

### 3.2 Where to get the facts

| Source | Use it for |
|---|---|
| Open Food Facts additives taxonomy | E-numbers, names, synonyms, usage frequency |
| EFSA re-evaluation opinions | ADI values, EU status — your best `info` source |
| FDA Substances Added to Food | US status, GRAS determinations |
| IARC Monographs | carcinogenicity classifications |
| WHO / JECFA | international ADIs |
| Regulation (EC) 1333/2008 Annexes | EU permitted list and warning requirements |

Start from the Open Food Facts taxonomy, sort additives by how often they occur
in real products, and take the top 40. That way your KB covers what people
actually scan.

### 3.3 Schema

Each entry:

```json
{
  "name": "Sodium benzoate",
  "synonyms": ["E211", "benzoate of soda", "benzoic acid", "E210"],
  "category": "preservative",
  "verdict": "care",
  "info": "A preservative for acidic foods and drinks. Intake limit of 5 mg per kg of body weight per day. In the presence of ascorbic acid it can form traces of benzene, which is why some drink recipes avoid the pairing.",
  "source": "EFSA Journal 2016;14(3):4433",
  "regulatory": {"eu": "permitted", "us": "permitted"}
}
```

Rules for `info`: two to three sentences, written by you, plain English, always
naming the number and the body that set it. Do not paste text from the source
document.

Rules for `synonyms`: include the E-number, the chemical name, the common name,
and any trade name. This list is what your retrieval actually searches, so it
matters more than the prose.

### 3.4 Starter `kb.json`

Here are five complete entries covering each rating so you have the pattern.
Extend to 40.

```json
[
  {
    "name": "Ascorbic acid",
    "synonyms": ["E300", "vitamin C", "sodium ascorbate", "L-ascorbic acid"],
    "category": "antioxidant",
    "verdict": "ok",
    "info": "Vitamin C, used to stop fats and colours oxidising. It is an essential nutrient, and EFSA concluded in its re-evaluation that no numerical intake limit was necessary.",
    "source": "EFSA Journal 2015;13(5):4087",
    "regulatory": {"eu": "permitted", "us": "GRAS"}
  },
  {
    "name": "Monosodium glutamate",
    "synonyms": ["MSG", "E621", "sodium glutamate", "glutamic acid", "E620"],
    "category": "flavour enhancer",
    "verdict": "care",
    "info": "A savoury flavour enhancer. EFSA's 2017 review set a group intake limit of 30 mg per kg of body weight per day and noted that some European consumers exceed it. The FDA classes it as generally recognised as safe.",
    "source": "EFSA Journal 2017;15(7):4910",
    "regulatory": {"eu": "permitted with ADI", "us": "GRAS"}
  },
  {
    "name": "Titanium dioxide",
    "synonyms": ["E171", "CI 77891"],
    "category": "colour",
    "verdict": "avoid",
    "info": "A white pigment. EFSA concluded in 2021 that it could no longer be considered safe as a food additive because genotoxicity could not be ruled out, and the EU banned it in food from 2022. It remains permitted in the United States.",
    "source": "EFSA Journal 2021;19(5):6585",
    "regulatory": {"eu": "banned 2022", "us": "permitted"}
  },
  {
    "name": "Partially hydrogenated oil",
    "synonyms": ["PHO", "partially hydrogenated vegetable oil", "hydrogenated vegetable oil", "trans fat", "shortening"],
    "category": "fat",
    "verdict": "avoid",
    "info": "The main industrial source of trans fat. It raises LDL cholesterol and lowers HDL. The FDA determined in 2015 that it is no longer generally recognised as safe, and the WHO has called for global elimination.",
    "source": "FDA Final Determination on PHOs, 2015",
    "regulatory": {"eu": "limited by Reg 2019/649", "us": "not GRAS"}
  },
  {
    "name": "Salt",
    "synonyms": ["sodium chloride", "sea salt", "rock salt", "iodised salt"],
    "category": "mineral",
    "verdict": "care",
    "info": "The WHO recommends adults consume less than 5 g of salt per day. Most intake in developed countries comes from processed food rather than added salt.",
    "source": "WHO Guideline on sodium intake, 2012",
    "regulatory": {"eu": "unrestricted", "us": "unrestricted"}
  }
]
```

Cover at minimum: sugar, salt, palm oil, PHO, HFCS, ascorbic acid, citric acid,
MSG, sodium benzoate, potassium sorbate, sodium nitrite, BHA, BHT, titanium
dioxide, erythrosine, tartrazine, sunset yellow, allura red, caramel colour,
aspartame, sucralose, acesulfame K, steviol glycosides, sorbitol, carrageenan,
xanthan gum, guar gum, pectin, soy lecithin, mono- and diglycerides, polysorbate
80, sodium bicarbonate, calcium carbonate, sodium phosphate, potassium bromate,
azodicarbonamide, propylparaben, maltodextrin, natural flavouring, yeast extract.

**Verify every citation against the primary source before you submit.** A wrong
ADI in a document that claims to be sourced is worse than no number at all.

---

## Step 4 — `build_index.py`

Run once. Converts `kb.json` into a Chroma vector store.

```python
"""Build the vector store from kb.json. Run once, or after editing the KB."""

import json
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"


def load_kb(path="kb.json"):
    with open(path, encoding="utf-8") as f:
        kb = json.load(f)

    seen = set()
    for entry in kb:
        for field in ("name", "synonyms", "verdict", "info", "source"):
            if field not in entry:
                raise ValueError(f"{entry.get('name', '?')} is missing '{field}'")
        if entry["verdict"] not in ("ok", "care", "avoid"):
            raise ValueError(f"{entry['name']} has an invalid verdict")
        key = entry["name"].lower()
        if key in seen:
            raise ValueError(f"duplicate entry: {entry['name']}")
        seen.add(key)

    return kb


def to_documents(kb):
    """One document per ingredient.

    Name and synonyms go into the embedded text so that a label reading
    'E621' and one reading 'monosodium glutamate' both retrieve the
    same record.
    """
    docs = []
    for e in kb:
        text = (
            f"{e['name']}. "
            f"Also known as: {', '.join(e['synonyms'])}. "
            f"Category: {e['category']}. "
            f"{e['info']}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "name": e["name"],
                    "verdict": e["verdict"],
                    "category": e["category"],
                    "source": e["source"],
                    "synonyms": ", ".join(e["synonyms"]),
                },
            )
        )
    return docs


def main():
    kb = load_kb()
    docs = to_documents(kb)

    if Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    counts = {}
    for e in kb:
        counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1

    print(f"Indexed {len(docs)} ingredients into {PERSIST_DIR}")
    print(f"Ratings: {counts}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python build_index.py
```

Expected output:

```
Indexed 40 ingredients into ./chroma_db
Ratings: {'ok': 14, 'care': 18, 'avoid': 8}
```

The embedding model downloads on first run, about 90 MB. It runs locally, so
there is no API cost for embeddings.

---

## Step 5 — `barcode_detector.py`

```python
"""Read text from a label photo."""

import easyocr

_reader = None


def get_reader():
    """Load lazily — model init takes a few seconds."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text(image_path, min_conf=0.3):
    """Return (text, mean_confidence).

    Lines below min_conf are dropped as noise before the mean is taken.
    """
    results = get_reader().readtext(image_path)

    kept = [(text, conf) for _, text, conf in results if conf >= min_conf]
    if not kept:
        return "", 0.0

    text = " ".join(t for t, _ in kept)
    mean_conf = sum(c for _, c in kept) / len(kept)
    return text, mean_conf
```

Test it on any label photo:

```bash
python -c "from barcode_detector import extract_text; print(extract_text('images/test1.jpg'))"
```

---

## Step 6 — `chain.py`

The core. Two LLM calls with retrieval between them.

```python
"""RAG pipeline: OCR text → parsed ingredients → retrieval → explanation."""

import json
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Check console.anthropic.com for current model names before submitting.
MODEL = "claude-sonnet-5"
DISTANCE_THRESHOLD = 1.0   # Chroma L2: lower is a closer match

llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=2000)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


# ------------------------------------------------------------------
# Chain 1 — parse the raw OCR text into a clean ingredient list
# ------------------------------------------------------------------

PARSE_PROMPT = ChatPromptTemplate.from_template("""
You are extracting the ingredient list from OCR output of a food label.

Rules:
1. Fix only obvious character-level OCR errors: 0 read as O, 1 as l, 5 as S.
2. Split bracketed sub-ingredients into separate entries.
   "raising agents (sodium bicarbonate, E503)" becomes three entries:
   raising agents, sodium bicarbonate, E503.
3. Drop text after "Contains", "Allergy advice", "Nutrition", "Best before".
4. Lowercase everything. Strip percentages from the name.
5. If a token is unrecognisable, leave it out. NEVER invent an ingredient
   that is not in the text.

Return ONLY a JSON array of strings. No preamble, no markdown fences.

OCR text:
{ocr_text}
""")

parse_chain = PARSE_PROMPT | llm | JsonOutputParser()


def parse_ingredients(ocr_text):
    """Return a list of cleaned ingredient names."""
    try:
        result = parse_chain.invoke({"ocr_text": ocr_text})
        if isinstance(result, list):
            return [str(i).strip().lower() for i in result if str(i).strip()]
        return []
    except Exception as exc:
        print(f"[parse] LLM failed ({exc}), falling back to comma split")
        cleaned = ocr_text.lower()
        for stop in ("contains", "allergy advice", "nutrition", "best before"):
            cleaned = cleaned.split(stop)[0]
        if "ingredients" in cleaned:
            cleaned = cleaned.split("ingredients", 1)[1].lstrip(": ")
        return [
            p.strip(" .()")
            for p in cleaned.replace("(", ",").replace(")", ",").split(",")
            if len(p.strip(" .()")) > 1
        ]


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------

def retrieve(ingredients):
    """Look up each ingredient. Returns (context_string, verdict_map, matches)."""
    blocks = []
    verdicts = {}
    matches = {}

    for ing in ingredients:
        hits = store.similarity_search_with_score(ing, k=1)

        if not hits or hits[0][1] > DISTANCE_THRESHOLD:
            verdicts[ing] = "unknown"
            matches[ing] = None
            continue

        doc, distance = hits[0]
        meta = doc.metadata

        verdicts[ing] = meta["verdict"]
        matches[ing] = {
            "name": meta["name"],
            "category": meta["category"],
            "source": meta["source"],
            "distance": round(float(distance), 3),
        }

        blocks.append(
            f"LABEL TEXT: {ing}\n"
            f"MATCHED RECORD: {meta['name']} ({meta['category']})\n"
            f"RATING: {meta['verdict']}\n"
            f"EVIDENCE: {doc.page_content}\n"
            f"SOURCE: {meta['source']}"
        )

    return "\n\n---\n\n".join(blocks), verdicts, matches


# ------------------------------------------------------------------
# Chain 2 — write the explanation, grounded in retrieved context
# ------------------------------------------------------------------

EXPLAIN_PROMPT = ChatPromptTemplate.from_template("""
You are a food ingredient analyst writing for an ordinary shopper.

Use ONLY the context below. Do not add facts from your own knowledge.

Hard rules:
- Every claim must come from the context, and you must name the SOURCE.
- The RATING in the context is fixed. Explain it; never change it.
- Do not say a product is safe or unsafe for any person or condition.
- Do not give medical, dietary, or treatment advice.
- For ingredients listed as not found, say plainly that you have no
  information. Do not guess.

Write:
1. One sentence per rated ingredient explaining its rating, with its source.
2. A two-sentence overall summary of the label.
3. A line noting that ratings describe the substance, not the amount present.

CONTEXT
{context}

INGREDIENTS ON THE LABEL
{ingredients}

NOT FOUND IN THE DATABASE
{unknowns}
""")

explain_chain = EXPLAIN_PROMPT | llm | StrOutputParser()


# ------------------------------------------------------------------
# Guardrail
# ------------------------------------------------------------------

BANNED_PHRASES = [
    "cures", "will prevent", "treats your", "safe for you",
    "you should stop eating", "diagnos", "prescrib",
]


def check_output(text):
    """Flag medical-claim language. Returns list of violations."""
    lowered = text.lower()
    return [p for p in BANNED_PHRASES if p in lowered]


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

def analyse_label(ocr_text, skip_llm=False):
    """Full pipeline. Set skip_llm=True for the retrieval-only baseline."""

    ingredients = parse_ingredients(ocr_text)
    context, verdicts, matches = retrieve(ingredients)
    unknowns = [i for i, v in verdicts.items() if v == "unknown"]

    result = {
        "ingredients": ingredients,
        "verdicts": verdicts,      # from KB metadata, never from the LLM
        "matches": matches,
        "unknowns": unknowns,
        "explanation": None,
        "violations": [],
    }

    if skip_llm:
        return result

    try:
        explanation = explain_chain.invoke({
            "context": context or "No ingredients matched the database.",
            "ingredients": ", ".join(ingredients),
            "unknowns": ", ".join(unknowns) or "None",
        })
        violations = check_output(explanation)
        if violations:
            explanation = (
                "The generated explanation was withheld because it contained "
                "language that reads as medical advice. Ratings above are "
                "unaffected."
            )
        result["explanation"] = explanation
        result["violations"] = violations
    except Exception as exc:
        # Degrade gracefully: ratings still work without the LLM
        result["explanation"] = f"(Explanation unavailable: {exc})"

    return result


if __name__ == "__main__":
    sample = (
        "Ingredients: wheat flour, sugar, palm oil, cocoa powder (4.5%), "
        "raising agents (sodium bicarbonate, E503), salt, "
        "emulsifier (soy lecithin), natural flavouring."
    )
    out = analyse_label(sample)
    print(json.dumps(out["verdicts"], indent=2))
    print()
    print(out["explanation"])
```

Test it:

```bash
python chain.py
```

### The design point to emphasise in your report

`verdicts` is read from Chroma metadata. The LLM never produces it. This means:

- the same label always gives the same rating
- a hallucinated health claim cannot change a rating
- updating `kb.json` and re-indexing changes behaviour with no retraining

The LLM does two things it is genuinely good at — repairing messy text and
writing readable prose — and nothing it is bad at.

---

## Step 7 — `app.py`

```python
"""Streamlit UI."""

import tempfile

import streamlit as st
from PIL import Image

from chain import analyse_label
from barcode_detector import extract_text

st.set_page_config(page_title="Ingredient Checker", page_icon="🥫")

BADGE = {
    "ok": ("🟢", "Fine"),
    "care": ("🟡", "Use with care"),
    "avoid": ("🔴", "Avoid"),
    "unknown": ("⚪", "Not in database"),
}

SAMPLES = {
    "Chocolate cookies": (
        "Ingredients: Wheat flour, sugar, palm oil, cocoa powder (4.5%), "
        "invert syrup, raising agents (sodium bicarbonate, ammonium "
        "bicarbonate), salt, emulsifier (soy lecithin), natural flavouring."
    ),
    "Diet cola": (
        "Ingredients: Carbonated water, caramel colour (E150d), phosphoric "
        "acid, aspartame, potassium benzoate, natural flavourings, citric "
        "acid, acesulfame K, caffeine."
    ),
    "Deli ham": (
        "Ingredients: Pork (92%), water, salt, dextrose, sodium phosphate, "
        "sodium ascorbate, sodium nitrite, natural flavouring."
    ),
}

st.title("🥫 Ingredient Checker")
st.caption(
    "Ratings describe published regulatory assessments of each substance. "
    "They do not account for how much is in the product and are not health advice."
)

tab_text, tab_photo = st.tabs(["Paste text", "Scan a photo"])
label_text = ""

with tab_text:
    choice = st.selectbox("Load a sample", ["—"] + list(SAMPLES))
    default = SAMPLES.get(choice, "")
    label_text = st.text_area("Ingredient list", default, height=140)

with tab_photo:
    photo = st.file_uploader("Label photo", type=["jpg", "jpeg", "png"])
    if photo:
        st.image(Image.open(photo), width=320)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(photo.getvalue())
            path = tmp.name

        with st.spinner("Reading the label…"):
            scanned, conf = extract_text(path)

        st.caption(f"Read at {conf:.0%} confidence")
        if conf < 0.6:
            st.warning(
                "Confidence is low. Retake the photo closer and flatter, "
                "or correct the text below."
            )
        label_text = st.text_area("Correct any mistakes", scanned, height=140)

if st.button("Check label", type="primary"):
    if not label_text.strip():
        st.error("Add an ingredient list first.")
    else:
        with st.spinner("Checking…"):
            result = analyse_label(label_text)

        counts = {k: 0 for k in BADGE}
        for v in result["verdicts"].values():
            counts[v] += 1

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fine", counts["ok"])
        c2.metric("Use with care", counts["care"])
        c3.metric("Avoid", counts["avoid"])
        c4.metric("Unknown", counts["unknown"])

        st.subheader("Ingredients")
        for ing, verdict in result["verdicts"].items():
            icon, word = BADGE[verdict]
            match = result["matches"].get(ing)
            if match:
                st.markdown(
                    f"{icon} **{ing}** — {word}  \n"
                    f"<small>matched to {match['name']} · {match['category']} · "
                    f"{match['source']}</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"{icon} **{ing}** — {word}")

        if result["explanation"]:
            st.subheader("What this means")
            st.write(result["explanation"])

        if result["unknowns"]:
            st.info(
                f"Not in the database: {', '.join(result['unknowns'])}. "
                "These are reported as unknown rather than guessed at."
            )
```

Run it:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Step 8 — Evaluation

Skipping this is the most common way to lose marks on a RAG assignment.

### 8.1 `test_labels.json`

Ten labels, hand-written, with expected ratings. Mix in deliberate OCR-style
noise so you are testing the repair path too.

```json
[
  {
    "id": "cookies_clean",
    "text": "Ingredients: wheat flour, sugar, palm oil, salt, soy lecithin, ascorbic acid",
    "expected": {
      "wheat flour": "ok",
      "sugar": "care",
      "palm oil": "care",
      "salt": "care",
      "soy lecithin": "ok",
      "ascorbic acid": "ok"
    }
  },
  {
    "id": "cookies_ocr_noise",
    "text": "INGREDIENTS: WHEAT FL0UR, SUGAR, PALM 0IL, SALT, S0Y LECITHlN, ASC0RBIC ACID",
    "expected": {
      "wheat flour": "ok",
      "sugar": "care",
      "palm oil": "care",
      "salt": "care",
      "soy lecithin": "ok",
      "ascorbic acid": "ok"
    }
  },
  {
    "id": "cola_enumbers",
    "text": "Ingredients: carbonated water, E150d, aspartame, E211, citric acid, E950",
    "expected": {
      "carbonated water": "ok",
      "e150d": "care",
      "aspartame": "care",
      "e211": "care",
      "citric acid": "ok",
      "e950": "care"
    }
  },
  {
    "id": "nested_brackets",
    "text": "Ingredients: flour, raising agents (sodium bicarbonate, E503), emulsifier (soy lecithin)",
    "expected": {
      "flour": "ok",
      "sodium bicarbonate": "ok",
      "e503": "ok",
      "soy lecithin": "ok"
    }
  },
  {
    "id": "contains_avoid",
    "text": "Ingredients: pork, salt, sodium nitrite, titanium dioxide, BHA",
    "expected": {
      "salt": "care",
      "sodium nitrite": "avoid",
      "titanium dioxide": "avoid",
      "bha": "avoid"
    }
  },
  {
    "id": "unknown_handling",
    "text": "Ingredients: quinoa flour, baobab powder, sugar, salt",
    "expected": {
      "quinoa flour": "unknown",
      "baobab powder": "unknown",
      "sugar": "care",
      "salt": "care"
    }
  }
]
```

Write ten. Include at least one that is entirely unknown ingredients — that
tests the most important guardrail.

### 8.2 `evaluate.py`

```python
"""Evaluation harness: accuracy, hallucination rate, ablations."""

import json
import time

from chain import analyse_label, parse_ingredients, retrieve


def load_tests(path="test_labels.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_one(test, skip_llm=False):
    """Run one label, return per-test metrics."""
    result = analyse_label(test["text"], skip_llm=skip_llm)
    verdicts = result["verdicts"]
    source = test["text"].lower()

    correct = 0
    total = 0
    misses = []

    for ing, expected in test["expected"].items():
        total += 1
        # match on the parsed key, allowing for minor formatting differences
        got = verdicts.get(ing)
        if got is None:
            for k, v in verdicts.items():
                if ing in k or k in ing:
                    got = v
                    break
        if got == expected:
            correct += 1
        else:
            misses.append((ing, expected, got))

    # A hallucinated ingredient is one the model produced that does not
    # appear anywhere in the input text.
    hallucinated = [
        i for i in result["ingredients"]
        if not any(word in source for word in i.split() if len(word) > 3)
    ]

    return {
        "id": test["id"],
        "correct": correct,
        "total": total,
        "misses": misses,
        "hallucinated": hallucinated,
        "unknown_count": len(result["unknowns"]),
    }


def run(skip_llm=False, label=""):
    tests = load_tests()
    start = time.time()

    correct = total = 0
    all_misses = []
    all_halluc = []

    for t in tests:
        r = score_one(t, skip_llm=skip_llm)
        correct += r["correct"]
        total += r["total"]
        all_misses += [(r["id"], *m) for m in r["misses"]]
        all_halluc += r["hallucinated"]

    elapsed = time.time() - start

    print(f"\n{'=' * 56}")
    print(f"  {label}")
    print("=" * 56)
    print(f"Rating accuracy      {correct}/{total} = {correct / total:.1%}")
    print(f"Hallucinated items   {len(all_halluc)}")
    print(f"Time per label       {elapsed / len(tests):.2f}s")

    if all_misses:
        print("\nMisclassified:")
        for test_id, ing, exp, got in all_misses:
            print(f"  [{test_id}] {ing}: expected {exp}, got {got}")

    if all_halluc:
        print(f"\nHallucinated: {all_halluc}")

    return {
        "accuracy": correct / total,
        "hallucinations": len(all_halluc),
        "seconds_per_label": elapsed / len(tests),
    }


def retrieval_recall(k_values=(1, 3, 5)):
    """Does the correct record appear in the top k?"""
    tests = load_tests()
    print(f"\n{'=' * 56}")
    print("  Retrieval recall@k")
    print("=" * 56)

    from chain import store

    for k in k_values:
        hits = total = 0
        for t in tests:
            for ing, expected in t["expected"].items():
                if expected == "unknown":
                    continue
                total += 1
                docs = store.similarity_search(ing, k=k)
                if any(d.metadata["verdict"] == expected for d in docs):
                    hits += 1
        print(f"  recall@{k}  {hits}/{total} = {hits / total:.1%}")


def threshold_sweep():
    """Does the unknown cut-off matter? Shows the precision/recall trade-off."""
    import chain

    tests = load_tests()
    print(f"\n{'=' * 56}")
    print("  Distance threshold sweep")
    print("=" * 56)

    original = chain.DISTANCE_THRESHOLD
    for th in (0.6, 0.8, 1.0, 1.2, 1.5):
        chain.DISTANCE_THRESHOLD = th
        correct = total = unknown = 0
        for t in tests:
            r = analyse_label(t["text"], skip_llm=True)
            for ing, expected in t["expected"].items():
                total += 1
                got = r["verdicts"].get(ing)
                if got == expected:
                    correct += 1
                if got == "unknown":
                    unknown += 1
        print(f"  threshold {th}:  accuracy {correct / total:.1%}, "
              f"marked unknown {unknown}/{total}")
    chain.DISTANCE_THRESHOLD = original


if __name__ == "__main__":
    baseline = run(skip_llm=True, label="ABLATION A — retrieval only, no LLM")
    full = run(skip_llm=False, label="ABLATION B — full pipeline with LLM")

    retrieval_recall()
    threshold_sweep()

    print(f"\n{'=' * 56}")
    print("  SUMMARY")
    print("=" * 56)
    print(f"  Retrieval only:  {baseline['accuracy']:.1%} accuracy, "
          f"{baseline['seconds_per_label']:.2f}s per label")
    print(f"  With LLM parse:  {full['accuracy']:.1%} accuracy, "
          f"{full['seconds_per_label']:.2f}s per label")
    print(f"  Hallucinations:  {full['hallucinations']}")
```

Run it:

```bash
python evaluate.py
```

### 8.3 What the results tell you

You now have three findings to write up:

1. **Does the LLM parsing step earn its cost?** Compare ablation A and B. It
   should help most on the OCR-noise test cases and matter little on clean text.
   Report the latency and cost difference too.
2. **Is the unknown threshold set correctly?** The sweep shows the trade-off:
   loose thresholds raise accuracy but produce confident wrong matches; tight
   ones mark too much as unknown. Pick a value and justify it.
3. **Hallucination count.** Should be zero. Stating that you measured it, and
   how, is the point.

Put these in a table in your report. A table with real numbers from your own
runs is worth more than three pages of description.

---

## Step 9 — Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` on a langchain import | The package layout moved. Check current docs for the module path; import errors are usually a rename, not a missing install. |
| Chroma returns stale results after editing `kb.json` | `build_index.py` deletes and rebuilds, so just re-run it. |
| Every ingredient comes back unknown | Threshold too tight, or you never ran `build_index.py`. Print the raw distances to see the actual range. |
| Everything matches something | Threshold too loose. Run the sweep and look at where wrong matches start appearing. |
| `JsonOutputParser` errors | The model wrapped output in markdown fences. The fallback in `parse_ingredients` covers this; you can also strip fences before parsing. |
| EasyOCR install fails | It needs PyTorch. On low-spec machines, skip OCR entirely and demo the paste path — say so in the report. |
| API authentication error | `.env` not loaded, or the key has a typo. Confirm with `python -c "import os,dotenv;dotenv.load_dotenv();print(os.getenv('ANTHROPIC_API_KEY')[:12])"`. |
| Streamlit reruns everything on each click | Normal. Cache the reader with `@st.cache_resource` if it is slow. |

---

## Step 10 — Report structure

Roughly 8–10 pages.

**1. Problem and motivation** (½ page)
What the app does and who for. State plainly that "healthy vs harmful" is not a
scientific binary, and that you are rating substances against published
regulatory assessments rather than issuing health verdicts. Framing this
correctly up front signals that you understood the problem.

**2. Why RAG rather than fine-tuning or a bare LLM** (1 page)
Three arguments: regulatory data changes and a KB edit is cheaper than
retraining; every rating traces to a citation; an unknown ingredient produces
"not in database" instead of a plausible fabrication. Note that a bare LLM asked
"is sodium benzoate harmful" gives fluent, unciteable, non-reproducible output.

**3. System architecture** (1½ pages)
Draw the two pipelines — offline KB construction, online serving. Walk through
the six stages. Include the data schema.

**4. Knowledge base construction** (1½ pages)
Your sources, your rating rubric written out explicitly, and a worked example of
applying it. Discuss the titanium dioxide case: banned in the EU, permitted in
the US. Explain which way you ruled and why. This paragraph shows judgement.

**5. Implementation** (1½ pages)
Key code, not all of it. Highlight the rating/explanation split and explain why
the LLM is deliberately excluded from producing ratings.

**6. Evaluation** (2 pages)
Method, gold set design, results tables, the two ablations, the threshold sweep.
Discuss the failure cases individually — a paragraph analysing why one
ingredient was misclassified is worth more than another chart.

**7. Limitations** (1 page)
Be direct. No dose awareness — the app cannot tell 0.1% sugar from 60%. Small
KB. Single-region ratings. OCR fails on curved and glossy packaging. The
regulatory disagreement problem has no clean answer. Fuzzy matching produces
occasional silent mismatches.

**8. Future work** (½ page)
Hybrid BM25 + dense retrieval with reciprocal rank fusion; cross-encoder
reranking; barcode lookup against Open Food Facts as an exact-match fast path;
quantity-aware ratings using the nutrition panel; a review queue that turns
unknown ingredients into KB growth.

---

## Demo script

Five minutes, in this order:

1. Paste the cookie sample → clean result, mostly green and amber
2. Paste the deli ham sample → sodium nitrite in red, read out the citation
3. Paste an ingredient the KB does not contain → show it returns unknown, and
   say why that matters more than getting it right
4. Upload a photo → show the confidence score and the editable correction box
5. Open `kb.json`, change one rating, re-run `build_index.py`, re-run the same
   label → the output changes with no model retraining. That is the RAG argument
   in ten seconds.

---

## Checklist before submitting

- [ ] `kb.json` has 40+ entries, every one with a real verified citation
- [ ] Rating rubric written out in the report
- [ ] `build_index.py` runs clean from an empty `chroma_db/`
- [ ] `evaluate.py` produces numbers you have pasted into the report
- [ ] Both ablations run and are discussed, not just printed
- [ ] Hallucination count reported explicitly
- [ ] At least one failure case analysed in prose
- [ ] Limitations section is honest about dose-blindness
- [ ] `.env` is not in the repository
- [ ] `requirements.txt` has pinned versions that actually worked
- [ ] README with setup commands in order
