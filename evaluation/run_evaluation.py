"""
Evaluation runner for AI Meeting Assistant.

Runs test cases against prompts and logs results to LangSmith.

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --suite summary_quality
    python evaluation/run_evaluation.py --prompt-version 1.0.0
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import InferenceClient
from prompts import get_prompt, EXECUTIVE_SUMMARY, TASK_EXTRACTION_MULTI_SPEAKER, TASK_EXTRACTION_SINGLE_SPEAKER

HF_TOKEN = os.environ.get("HF_TOKEN")
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"

TEST_CASES_DIR = Path(__file__).parent / "test_cases"
RESULTS_DIR = Path(__file__).parent / "results"


def load_test_suite(suite_name: str) -> dict:
    path = TEST_CASES_DIR / f"{suite_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Test suite not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_available_suites() -> list:
    return [f.stem for f in TEST_CASES_DIR.glob("*.json")]


def call_llm(client, messages: list, model: str, max_tokens: int, temperature: float) -> str:
    response = client.chat_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choices = response.get("choices", [])
    if choices and choices[0].get("message", {}).get("content"):
        return choices[0]["message"]["content"].strip()
    return ""


def evaluate_summary(client, test_case: dict) -> dict:
    prompt = EXECUTIVE_SUMMARY
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"].format(text=test_case["input"])},
    ]

    start = time.monotonic()
    response = call_llm(client, messages, LLM_MODEL, prompt["max_tokens"], prompt["temperature"])
    latency_ms = int((time.monotonic() - start) * 1000)

    response_lower = response.lower()

    keyword_hits = sum(1 for kw in test_case.get("expected_keywords", []) if kw.lower() in response_lower)
    keyword_total = len(test_case.get("expected_keywords", []))
    keyword_score = keyword_hits / keyword_total if keyword_total > 0 else 1.0

    contains_hits = sum(1 for kw in test_case.get("expected_contains", []) if kw.lower() in response_lower)
    contains_total = len(test_case.get("expected_contains", []))
    contains_score = contains_hits / contains_total if contains_total > 0 else 1.0

    overall_score = (keyword_score * 0.6 + contains_score * 0.4)

    return {
        "response": response,
        "latency_ms": latency_ms,
        "keyword_score": round(keyword_score, 3),
        "contains_score": round(contains_score, 3),
        "overall_score": round(overall_score, 3),
        "passed": overall_score >= 0.7,
        "keyword_hits": keyword_hits,
        "keyword_total": keyword_total,
    }


def evaluate_task_extraction(client, test_case: dict) -> dict:
    is_single = test_case.get("speaker_count", 1) <= 1
    prompt = TASK_EXTRACTION_SINGLE_SPEAKER if is_single else TASK_EXTRACTION_MULTI_SPEAKER
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"].format(text=test_case["input"])},
    ]

    start = time.monotonic()
    response = call_llm(client, messages, LLM_MODEL, prompt["max_tokens"], prompt["temperature"])
    latency_ms = int((time.monotonic() - start) * 1000)

    response_lower = response.lower()

    expected_exact = test_case.get("expected_exact")
    if expected_exact and expected_exact.lower() in response_lower:
        return {
            "response": response,
            "latency_ms": latency_ms,
            "passed": True,
            "score": 1.0,
            "matched_expected_exact": True,
        }

    expected_tasks = test_case.get("expected_tasks", [])
    if expected_tasks:
        hits = sum(1 for t in expected_tasks if any(word in response_lower for word in t.lower().split()[:3]))
        score = hits / len(expected_tasks) if expected_tasks else 1.0
    else:
        no_tasks_phrase = "no actionable tasks identified" in response_lower
        score = 1.0 if no_tasks_phrase else 0.0

    return {
        "response": response,
        "latency_ms": latency_ms,
        "passed": score >= 0.5,
        "score": round(score, 3),
        "matched_expected_exact": False,
    }


def run_evaluation(suite_name: str = None, prompt_version: str = None) -> dict:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Set it in environment or HF Space secrets.")

    client = InferenceClient(token=HF_TOKEN)
    suites_to_run = [suite_name] if suite_name else get_available_suites()
    all_results = {}

    for suite_name in suites_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {suite_name}")
        print(f"{'='*60}")

        suite = load_test_suite(suite_name)
        results = {"test_suite": suite_name, "version": suite.get("version", "unknown"), "cases": []}

        for tc in suite["test_cases"]:
            print(f"  {tc['id']}...", end=" ", flush=True)

            if suite_name == "summary_quality":
                eval_result = evaluate_summary(client, tc)
            elif suite_name == "task_extraction":
                eval_result = evaluate_task_extraction(client, tc)
            else:
                print(f"SKIP (unknown suite type)")
                continue

            status = "PASS" if eval_result["passed"] else "FAIL"
            print(f"{status} ({eval_result['latency_ms']}ms)")

            results["cases"].append({
                "id": tc["id"],
                "category": tc.get("category"),
                "tags": tc.get("tags", []),
                **eval_result,
            })

            # Log to LangSmith
            try:
                from tracing import log_evaluation_result
                log_evaluation_result(
                    prompt_name=suite_name,
                    prompt_version=prompt_version or suite.get("version", "unknown"),
                    test_case_id=tc["id"],
                    transcript_preview=tc["input"][:200],
                    response=eval_result["response"][:200],
                    scores={k: v for k, v in eval_result.items() if k != "response"},
                )
            except Exception:
                pass

        passed = sum(1 for c in results["cases"] if c["passed"])
        total = len(results["cases"])
        avg_latency = sum(c["latency_ms"] for c in results["cases"]) / total if total else 0

        results["summary"] = {
            "passed": passed,
            "total": total,
            "accuracy": round(passed / total, 3) if total else 0,
            "avg_latency_ms": int(avg_latency),
        }

        print(f"\n  Summary: {passed}/{total} passed ({results['summary']['accuracy']*100:.1f}%)")
        print(f"  Avg latency: {avg_latency:.0f}ms")

        all_results[suite_name] = results

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version = prompt_version or "latest"
    output_path = RESULTS_DIR / f"{timestamp}_{version}.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run meeting assistant evaluation")
    parser.add_argument("--suite", type=str, default=None, help="Specific test suite to run")
    parser.add_argument("--prompt-version", type=str, default=None, help="Prompt version tag")
    args = parser.parse_args()

    results = run_evaluation(suite_name=args.suite, prompt_version=args.prompt_version)

    for suite_name, results in results.items():
        accuracy = results["summary"]["accuracy"]
        if accuracy < 0.7:
            print(f"\nWARNING: {suite_name} accuracy below threshold ({accuracy*100:.1f}% < 70%)")
            sys.exit(1)

    print("\nAll suites passed.")


if __name__ == "__main__":
    main()
