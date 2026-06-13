"""Tests for tracing module."""
import json
import pytest
from unittest.mock import patch, MagicMock
from tracing import (
    _TraceRun,
    trace_meeting_analysis,
    log_llm_call,
    log_evaluation_result,
    LANGSMITH_ENABLED,
)


class TestTraceRun:
    def test_log_step(self):
        run_data = {"steps": [], "total_latency_ms": 0}
        trace = _TraceRun(run_data)
        trace.log("transcription", latency_ms=1000, chars=500)
        assert len(run_data["steps"]) == 1
        assert run_data["steps"][0]["name"] == "transcription"
        assert run_data["steps"][0]["latency_ms"] == 1000

    def test_multiple_steps(self):
        run_data = {"steps": [], "total_latency_ms": 0}
        trace = _TraceRun(run_data)
        trace.log("step1", latency_ms=100)
        trace.log("step2", latency_ms=200)
        assert len(run_data["steps"]) == 2

    def test_steps_property(self):
        run_data = {"steps": [{"name": "test"}], "total_latency_ms": 0}
        trace = _TraceRun(run_data)
        assert trace.steps == [{"name": "test"}]


class TestTraceMeetingAnalysis:
    def test_context_manager_populates_data(self):
        with trace_meeting_analysis(60.0, False, "1.0.0") as trace:
            trace.log("test_step", latency_ms=500)
        # Should not raise

    def test_context_manager_with_error(self):
        with pytest.raises(ValueError):
            with trace_meeting_analysis(60.0, False, "1.0.0") as trace:
                raise ValueError("test error")


class TestLogLlmCall:
    @patch("tracing.get_client", return_value=None)
    def test_logs_without_langsmith(self, mock_client):
        # Should not raise even without LangSmith
        log_llm_call(
            prompt_name="test",
            prompt_version="1.0.0",
            model="test-model",
            input_text="test input",
            output_text="test output",
            latency_ms=100,
        )

    @patch("tracing.get_client", return_value=None)
    def test_logs_with_fallback(self, mock_client):
        log_llm_call(
            prompt_name="test",
            prompt_version="1.0.0",
            model="test-model",
            input_text="test input",
            output_text="test output",
            latency_ms=100,
            fallback_used="bart",
        )


class TestLogEvaluationResult:
    @patch("tracing.get_client", return_value=None)
    def test_logs_without_langsmith(self, mock_client):
        log_evaluation_result(
            prompt_name="test",
            prompt_version="1.0.0",
            test_case_id="tc_001",
            transcript_preview="test transcript",
            response="test response",
            scores={"accuracy": 4, "relevance": 5},
        )
