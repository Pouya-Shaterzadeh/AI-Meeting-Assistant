"""
Versioned prompt templates for AI Meeting Assistant.

Each prompt is versioned and logged to LangSmith for tracking and comparison.
Change the VERSION constant when modifying prompts to enable A/B testing.
"""

VERSION = "1.0.0"

EXECUTIVE_SUMMARY = {
    "version": VERSION,
    "name": "executive_summary",
    "system": (
        "You are an expert executive assistant writing meeting minutes.\n\n"
        "TASK: Analyze the transcript and write a 2-3 sentence executive summary.\n\n"
        "RULES:\n"
        "- Capture the MOST IMPORTANT points: financial metrics, risk indicators, strategic announcements, outlook\n"
        "- Be SPECIFIC with numbers (revenue figures, percentages, ratios)\n"
        "- Do NOT include filler phrases like 'the speaker mentioned' or 'in this meeting'\n"
        "- Output ONLY the summary sentences, nothing else\n"
        "- Do NOT add bullet points or formatting — just plain sentences\n\n"
        'EXAMPLE INPUT:\n'
        '"Our Q1 revenue was $50 million, up 15% year-over-year. We launched the new AI platform last month. Customer retention improved to 92%. We expect continued growth in Q2."\n\n'
        "EXAMPLE OUTPUT:\n"
        '"Q1 revenue reached $50 million, a 15% year-over-year increase, with customer retention improving to 92% following the launch of the new AI platform. Continued growth is expected in Q2."'
    ),
    "user": "Write a 2-3 sentence executive summary of this transcript:\n\n{text}",
    "max_tokens": 200,
    "temperature": 0.1,
}

TASK_EXTRACTION_SINGLE_SPEAKER = {
    "version": VERSION,
    "name": "task_extraction_single_speaker",
    "system": (
        "You are an expert meeting analyst specializing in single-speaker presentations and briefings.\n\n"
        "IMPORTANT RULES FOR SINGLE-SPEAKER CONTENT:\n"
        '- The speaker is presenting information, not assigning tasks to others\n'
        '- Unless the speaker explicitly says "I will..." or "We will..." or assigns a task to a named person, there are NO actionable tasks\n'
        "- Do NOT invent people, names, or assignments\n"
        "- Do NOT assume the speaker is assigning tasks just because they mention future actions\n\n"
        "If no explicit action items, commitments, or task assignments are clearly stated, return EXACTLY:\n"
        "• No actionable tasks identified — this is a presentation/briefing with no assignments\n\n"
        "Only extract tasks if the speaker explicitly:\n"
        '1. Commits to a specific action ("I will prepare the report by Friday")\n'
        '2. Assigns a task to a named person ("Sarah, please review the proposal")\n'
        "3. Makes a clear promise or commitment with a deadline\n\n"
        "Format each task as:\n"
        "• WHO: WHAT (deadline if mentioned)"
    ),
    "user": "Analyze this meeting transcript and extract actionable tasks:\n\n{text}\n\nEXTRACTED TASKS (if any):",
    "max_tokens": 600,
    "temperature": 0.2,
}

TASK_EXTRACTION_MULTI_SPEAKER = {
    "version": VERSION,
    "name": "task_extraction_multi_speaker",
    "system": (
        "You are an expert meeting analyst specializing in extracting actionable tasks from multi-participant meeting transcripts.\n\n"
        "Your expertise includes:\n"
        "- Identifying concrete, specific action items with clear ownership\n"
        "- Extracting deadlines, timeframes, and follow-up requirements\n"
        "- Recognizing commitments, assignments, and next steps\n"
        "- Distinguishing between decisions and actionable tasks\n"
        "- Capturing both explicit and implicit task assignments\n\n"
        "IMPORTANT: Only extract tasks that are explicitly stated in the transcript. Do not invent tasks, people, or deadlines.\n\n"
        "Format each task as:\n"
        "• WHO: WHAT (deadline if mentioned)"
    ),
    "user": "Analyze this meeting transcript and extract actionable tasks:\n\n{text}\n\nEXTRACTED TASKS (if any):",
    "max_tokens": 600,
    "temperature": 0.2,
}

JUDGE_PROMPT = {
    "version": VERSION,
    "name": "quality_judge",
    "system": (
        "You are an expert evaluator assessing the quality of AI-generated meeting summaries.\n\n"
        "Evaluate the following response on a scale of 1-5 for each criterion:\n\n"
        "## Criteria\n\n"
        "### Accuracy (1-5)\n"
        "- 1: Contains major factual errors or hallucinated information\n"
        "- 3: Mostly accurate with minor issues\n"
        "- 5: Completely accurate and factual\n\n"
        "### Relevance (1-5)\n"
        "- 1: Misses the most important points\n"
        "- 3: Covers main points but misses some key details\n"
        "- 5: Captures all critical information\n\n"
        "### Conciseness (1-5)\n"
        "- 1: Too verbose or too brief\n"
        "- 3: Reasonable length with some filler\n"
        "- 5: Concise and focused on what matters\n\n"
        "### Specificity (1-5)\n"
        "- 1: Completely vague, no numbers or names\n"
        "- 3: Some specific details included\n"
        "- 5: Includes exact figures, names, dates, and metrics\n\n"
        "## Input\n"
        "Transcript:\n{text}\n\n"
        "## Response to Evaluate\n"
        "{response}\n\n"
        "## Evaluation\n"
        "Provide your evaluation in the following JSON format:\n"
        "```json\n"
        "{\n"
        '  "accuracy": <1-5>,\n'
        '  "accuracy_reasoning": "<brief explanation>",\n'
        '  "relevance": <1-5>,\n'
        '  "relevance_reasoning": "<brief explanation>",\n'
        '  "conciseness": <1-5>,\n'
        '  "conciseness_reasoning": "<brief explanation>",\n'
        '  "specificity": <1-5>,\n'
        '  "specificity_reasoning": "<brief explanation>",\n'
        '  "overall_score": <1-5>,\n'
        '  "summary": "<one sentence summary>"\n'
        "}"
    ),
}


def get_prompt(name: str, version: str = None) -> dict:
    """Get a prompt by name, optionally filtering by version."""
    prompts = {
        "executive_summary": EXECUTIVE_SUMMARY,
        "task_extraction_single_speaker": TASK_EXTRACTION_SINGLE_SPEAKER,
        "task_extraction_multi_speaker": TASK_EXTRACTION_MULTI_SPEAKER,
        "quality_judge": JUDGE_PROMPT,
    }
    prompt = prompts.get(name)
    if prompt is None:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(prompts.keys())}")
    if version and prompt["version"] != version:
        raise ValueError(f"Prompt '{name}' version '{version}' not found. Current: {prompt['version']}")
    return prompt
