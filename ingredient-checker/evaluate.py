"""Evaluation harness: accuracy, hallucination rate, ablations."""

import json
import time

from chain import analyse_label


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


if __name__ == "__main__":
    baseline = run(skip_llm=True, label="ABLATION A — retrieval only, no LLM")
    full = run(skip_llm=False, label="ABLATION B — full pipeline with LLM")
    retrieval_recall()

    print(f"\n{'=' * 56}")
    print("  SUMMARY")
    print("=" * 56)
    print(f"  Retrieval only:  {baseline['accuracy']:.1%} accuracy, "
          f"{baseline['seconds_per_label']:.2f}s per label")
    print(f"  With LLM parse:  {full['accuracy']:.1%} accuracy, "
          f"{full['seconds_per_label']:.2f}s per label")
    print(f"  Hallucinations:  {full['hallucinations']}")
