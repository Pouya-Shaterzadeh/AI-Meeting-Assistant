"""Tests for prompts module."""
import pytest
from prompts import (
    EXECUTIVE_SUMMARY,
    TASK_EXTRACTION_SINGLE_SPEAKER,
    TASK_EXTRACTION_MULTI_SPEAKER,
    JUDGE_PROMPT,
    VERSION,
    get_prompt,
)


class TestPromptStructure:
    def test_executive_summary_has_required_fields(self):
        assert "version" in EXECUTIVE_SUMMARY
        assert "name" in EXECUTIVE_SUMMARY
        assert "system" in EXECUTIVE_SUMMARY
        assert "user" in EXECUTIVE_SUMMARY
        assert "max_tokens" in EXECUTIVE_SUMMARY
        assert "temperature" in EXECUTIVE_SUMMARY

    def test_task_extraction_single_speaker_has_required_fields(self):
        assert "version" in TASK_EXTRACTION_SINGLE_SPEAKER
        assert "name" in TASK_EXTRACTION_SINGLE_SPEAKER
        assert "system" in TASK_EXTRACTION_SINGLE_SPEAKER
        assert "user" in TASK_EXTRACTION_SINGLE_SPEAKER
        assert "max_tokens" in TASK_EXTRACTION_SINGLE_SPEAKER
        assert "temperature" in TASK_EXTRACTION_SINGLE_SPEAKER

    def test_task_extraction_multi_speaker_has_required_fields(self):
        assert "version" in TASK_EXTRACTION_MULTI_SPEAKER
        assert "name" in TASK_EXTRACTION_MULTI_SPEAKER
        assert "system" in TASK_EXTRACTION_MULTI_SPEAKER
        assert "user" in TASK_EXTRACTION_MULTI_SPEAKER
        assert "max_tokens" in TASK_EXTRACTION_MULTI_SPEAKER
        assert "temperature" in TASK_EXTRACTION_MULTI_SPEAKER

    def test_judge_prompt_has_required_fields(self):
        assert "version" in JUDGE_PROMPT
        assert "name" in JUDGE_PROMPT
        assert "system" in JUDGE_PROMPT


class TestPromptVersioning:
    def test_all_prompts_same_version(self):
        assert EXECUTIVE_SUMMARY["version"] == VERSION
        assert TASK_EXTRACTION_SINGLE_SPEAKER["version"] == VERSION
        assert TASK_EXTRACTION_MULTI_SPEAKER["version"] == VERSION
        assert JUDGE_PROMPT["version"] == VERSION

    def test_version_is_string(self):
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0


class TestPromptFormatting:
    def test_executive_summary_user_format(self):
        result = EXECUTIVE_SUMMARY["user"].format(text="Test transcript")
        assert "Test transcript" in result

    def test_task_extraction_single_user_format(self):
        result = TASK_EXTRACTION_SINGLE_SPEAKER["user"].format(text="Test transcript")
        assert "Test transcript" in result

    def test_task_extraction_multi_user_format(self):
        result = TASK_EXTRACTION_MULTI_SPEAKER["user"].format(text="Test transcript")
        assert "Test transcript" in result


class TestGetPrompt:
    def test_get_executive_summary(self):
        prompt = get_prompt("executive_summary")
        assert prompt == EXECUTIVE_SUMMARY

    def test_get_task_extraction_single(self):
        prompt = get_prompt("task_extraction_single_speaker")
        assert prompt == TASK_EXTRACTION_SINGLE_SPEAKER

    def test_get_task_extraction_multi(self):
        prompt = get_prompt("task_extraction_multi_speaker")
        assert prompt == TASK_EXTRACTION_MULTI_SPEAKER

    def test_get_judge_prompt(self):
        prompt = get_prompt("quality_judge")
        assert prompt == JUDGE_PROMPT

    def test_get_unknown_prompt_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt"):
            get_prompt("nonexistent")

    def test_get_prompt_wrong_version_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_prompt("executive_summary", version="99.99.99")


class TestPromptQuality:
    def test_summary_prompt_no_hallucination_instructions(self):
        system = EXECUTIVE_SUMMARY["system"]
        assert "Do NOT" in system or "do not" in system.lower()

    def test_single_speaker_explicit_tasks_only(self):
        system = TASK_EXTRACTION_SINGLE_SPEAKER["system"]
        assert "explicitly" in system.lower() or "only extract" in system.lower()

    def test_multi_speaker_no_invention(self):
        system = TASK_EXTRACTION_MULTI_SPEAKER["system"]
        assert "Do not invent" in system or "do not invent" in system.lower()
