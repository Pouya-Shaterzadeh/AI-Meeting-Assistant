"""
LangSmith tracing and structured metrics logging for AI Meeting Assistant.

Set these environment variables in HF Space secrets:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=your_langsmith_api_key
  LANGCHAIN_PROJECT=ai-meeting-assistant
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


def _is_langsmith_enabled():
    """Check at call time, not import time."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def _get_client():
    """Lazy-import LangSmith client to avoid startup cost when disabled."""
    if not _is_langsmith_enabled():
        return None
    try:
        from langsmith import Client
        return Client()
    except ImportError:
        logger.warning("langsmith not installed — tracing disabled")
        return None
    except Exception as e:
        logger.warning(f"LangSmith client init failed: {e}")
        return None


_client = None


def get_client():
    global _client
    if _client is None:
        _client = _get_client()
    return _client


@contextmanager
def trace_meeting_analysis(
    audio_duration: float,
    is_long_audio: bool,
    prompt_version: str,
):
    """
    Context manager that traces the full meeting analysis pipeline.

    Usage:
        with trace_meeting_analysis(56.0, False, "1.0.0") as trace:
            trace.log("transcription", latency_ms=3050, chars=788)
            trace.log("summary", latency_ms=1200, prompt_version="1.0.0")
    """
    run_data = {
        "audio_duration_s": audio_duration,
        "is_long_audio": is_long_audio,
        "prompt_version": prompt_version,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "steps": [],
        "total_latency_ms": 0,
    }

    start = time.monotonic()
    trace = _TraceRun(run_data)

    try:
        yield trace
    except Exception as e:
        run_data["error"] = str(e)
        raise
    finally:
        run_data["total_latency_ms"] = int((time.monotonic() - start) * 1000)
        run_data["end_time"] = datetime.now(timezone.utc).isoformat()
        _log_run(run_data)


class _TraceRun:
    """Internal helper to collect step data during a traced run."""

    def __init__(self, run_data: dict):
        self._data = run_data

    def log(self, step_name: str, **metrics):
        """Log a pipeline step with arbitrary metrics."""
        step = {"name": step_name, **metrics}
        self._data["steps"].append(step)
        self._data["total_latency_ms"] += metrics.get("latency_ms", 0)

    @property
    def steps(self):
        return self._data["steps"]


def _log_run(run_data: dict):
    """Send run data to LangSmith and local structured log."""
    # Structured log (always)
    logger.info(
        json.dumps(
            {
                "event": "meeting_analysis",
                "audio_duration_s": run_data.get("audio_duration_s"),
                "is_long_audio": run_data.get("is_long_audio"),
                "prompt_version": run_data.get("prompt_version"),
                "total_latency_ms": run_data.get("total_latency_ms"),
                "step_count": len(run_data.get("steps", [])),
                "error": run_data.get("error"),
                "steps": run_data.get("steps"),
            }
        )
    )

    # LangSmith trace
    client = get_client()
    if client is None:
        logger.info("LangSmith client not available — skipping remote trace")
        return

    try:
        project = os.environ.get("LANGCHAIN_PROJECT", "ai-meeting-assistant")

        client.create_run(
            name="meeting_analysis",
            run_type="chain",
            inputs={
                "audio_duration_s": run_data.get("audio_duration_s"),
                "is_long_audio": run_data.get("is_long_audio"),
                "prompt_version": run_data.get("prompt_version"),
            },
            outputs={
                "total_latency_ms": run_data.get("total_latency_ms"),
                "steps": run_data.get("steps", []),
                "error": run_data.get("error"),
            },
            extra={
                "start_time": run_data.get("start_time"),
                "end_time": run_data.get("end_time"),
            },
            project_name=project,
        )
        client.flush()
        logger.info("LangSmith trace sent successfully")
    except Exception as e:
        logger.warning(f"LangSmith trace failed: {e}")


def log_llm_call(
    prompt_name: str,
    prompt_version: str,
    model: str,
    input_text: str,
    output_text: str,
    latency_ms: int,
    temperature: float = 0.1,
    max_tokens: int = 200,
    fallback_used: Optional[str] = None,
):
    """
    Log a single LLM call for monitoring and evaluation.

    Call this after every LLM invocation (summary, task extraction, etc.)
    """
    record = {
        "event": "llm_call",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "model": model,
        "input_chars": len(input_text),
        "input_preview": input_text[:200] + "..." if len(input_text) > 200 else input_text,
        "output_chars": len(output_text),
        "output_preview": output_text[:200] + "..." if len(output_text) > 200 else output_text,
        "latency_ms": latency_ms,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "fallback_used": fallback_used,
    }
    logger.info(json.dumps(record))

    client = get_client()
    if client is None:
        return

    try:
        project = os.environ.get("LANGCHAIN_PROJECT", "ai-meeting-assistant")

        client.create_run(
            name=f"llm_{prompt_name}",
            run_type="llm",
            inputs={"text": input_text[:1000], "prompt_version": prompt_version},
            outputs={"response": output_text[:1000], "latency_ms": latency_ms},
            extra={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "fallback_used": fallback_used,
            },
            project_name=project,
        )
        client.flush()
    except Exception as e:
        logger.warning(f"LangSmith log_llm_call failed: {e}")


def log_evaluation_result(
    prompt_name: str,
    prompt_version: str,
    test_case_id: str,
    transcript_preview: str,
    response: str,
    scores: dict,
):
    """
    Log an evaluation result for prompt quality tracking.

    Use with the LLM-as-Judge to track prompt quality over time.
    """
    record = {
        "event": "evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "test_case_id": test_case_id,
        "transcript_preview": transcript_preview[:200],
        "response_preview": response[:200],
        "scores": scores,
    }
    logger.info(json.dumps(record))

    client = get_client()
    if client is None:
        return

    try:
        project = os.environ.get("LANGCHAIN_PROJECT", "ai-meeting-assistant")

        client.create_run(
            name=f"eval_{prompt_name}",
            run_type="chain",
            inputs={
                "test_case_id": test_case_id,
                "transcript": transcript_preview[:1000],
            },
            outputs={"scores": scores, "response": response[:1000]},
            extra={"prompt_version": prompt_version},
            project_name=project,
        )
        client.flush()
    except Exception as e:
        logger.warning(f"LangSmith log_evaluation_result failed: {e}")
