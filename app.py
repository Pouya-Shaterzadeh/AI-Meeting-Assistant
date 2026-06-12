import gradio as gr
import os
import tempfile
from datetime import datetime
import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import wave
import struct
import json
import hashlib
import math

# Monkey-patch gradio_client to fix JSON schema bug with additionalProperties: true
import gradio_client.utils as _gc_utils
import logging
_gc_logger = logging.getLogger("gradio_client_patch")
_orig_json_schema_to_python_type = _gc_utils.json_schema_to_python_type
def _safe_json_schema_to_python_type(schema):
    if not isinstance(schema, dict):
        _gc_logger.warning(f"json_schema_to_python_type got non-dict schema: {type(schema).__name__}={schema}")
        return "any"
    return _orig_json_schema_to_python_type(schema)
_gc_utils.json_schema_to_python_type = _safe_json_schema_to_python_type

_orig_json_schema_to_python_type_priv = _gc_utils._json_schema_to_python_type
def _safe_json_schema_to_python_type_priv(schema, defs):
    if not isinstance(schema, dict):
        _gc_logger.warning(f"_json_schema_to_python_type got non-dict schema: {type(schema).__name__}={schema}")
        return "str"
    return _orig_json_schema_to_python_type_priv(schema, defs)
_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type_priv



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face Inference API client (serverless — no local models)
from huggingface_hub import InferenceClient

# Optional: LangChain prompt templates (lightweight, no model loading)
try:
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
    logger.info("✅ LangChain loaded successfully")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("⚠️ LangChain not available - using simple text processing")

class MeetingAssistant:
    """AI Meeting Assistant powered by Hugging Face Inference API (serverless)"""

    # Model IDs — optimized for meeting analysis
    ASR_MODEL = "openai/whisper-large-v3-turbo"
    SUM_MODEL = "philschmid/bart-large-cnn-samsum"
    SENT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
    LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
    
    # Chunked processing constants
    CHUNK_DURATION = 30  # seconds per audio chunk
    CHUNK_OVERLAP = 5    # seconds overlap between chunks
    MAX_AUDIO_DURATION = 3600  # 60 minutes max
    CHECKPOINT_DIR = "/tmp/meeting_assistant_checkpoints"
    


    def __init__(self):
        """Initialize HF Inference client and prompt templates"""
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            logger.warning("⚠️ HF_TOKEN not set — Inference API calls will fail")
            self.client = None
        else:
            self.client = InferenceClient(token=self.hf_token)
            logger.info("✅ HF Inference API client initialized")

        # Initialize LangChain ChatPromptTemplate for task extraction
        self._init_langchain_chains()

        # Token budget for LLM task extraction
        self._llm_max_tokens = 600

        logger.info("🚀 AI Meeting Assistant initialized (serverless — HF Inference API)")
    
    def _init_langchain_chains(self):
        """Initialize advanced ChatPromptTemplate chains following documentation"""
        try:
            if LANGCHAIN_AVAILABLE:
                # Advanced Task Extraction Chain - following documentation structure
                self.task_extraction_template = ChatPromptTemplate.from_messages([
                    ("system", """
                    You are an expert meeting analyst specializing in extracting actionable tasks from meeting transcripts.
                    
                    Your expertise includes:
                    - Identifying concrete, specific action items with clear ownership
                    - Extracting deadlines, timeframes, and follow-up requirements
                    - Recognizing commitments, assignments, and next steps
                    - Distinguishing between decisions and actionable tasks
                    - Capturing both explicit and implicit task assignments
                    
                    Always provide structured, actionable output that participants can immediately act upon.
                    Focus on WHO needs to do WHAT by WHEN.
                    """),
                    ("human", """
                    Analyze this meeting transcript and extract ALL actionable tasks, assignments, and follow-ups:
                    
                    MEETING CONTEXT:
                    {meeting_text}
                    
                    EXTRACT AND FORMAT:
                    1. Explicit task assignments ("John will prepare the report")
                    2. Action items with deadlines ("Complete by Friday")
                    3. Follow-up requirements ("Schedule follow-up meeting")
                    4. Commitments and promises made during the meeting
                    5. Next steps and implementation tasks
                    6. Review and approval tasks
                    7. Communication and coordination tasks
                    
                    FORMAT REQUIREMENTS:
                    - Each task as a bullet point starting with '•'
                    - Include WHO is responsible when mentioned
                    - Include WHEN (deadline/timeframe) when mentioned
                    - Include WHAT (specific action) clearly
                    - Be specific and actionable
                    - Avoid vague or general statements
                    
                    EXTRACTED TASKS:
                    """)
                ])
                
                logger.info("✅ Advanced ChatPromptTemplate chains initialized")
            else:
                logger.warning("⚠️ LangChain not available - using fallback methods")
                
        except Exception as e:
            logger.error(f"Error initializing LangChain chains: {e}")
    
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio via HF Inference API (whisper-large-v3-turbo)"""
        try:
            if audio_path is None:
                return "Please upload an audio file.", []

            if not self.client:
                return "HF_TOKEN not configured. Set your Hugging Face read token in the Space settings.", []

            logger.info("⚡ Transcribing via HF Inference API (whisper-large-v3-turbo)...")

            result = self.client.automatic_speech_recognition(
                audio_path,
                model=self.ASR_MODEL
            )
            if isinstance(result, dict):
                transcription = result.get("text", "")
            elif isinstance(result, str):
                transcription = result
            else:
                transcription = ""

            if not transcription:
                return "No speech detected in the audio file.", []

            logger.info("✅ Transcription completed via HF Inference API")
            return transcription, []

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return f"Error transcribing audio: {str(e)}", []
    
    def extract_action_items(self, text, is_single_speaker=False):
        """Extract action items via LLM (Phi-3-mini) or fall back to regex"""
        try:
            if self.client:
                return self._extract_tasks_with_llm(text, is_single_speaker)
            else:
                return self._extract_tasks_fallback(text)
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
            return self._extract_tasks_fallback(text)

    def _extract_tasks_with_llm(self, text, is_single_speaker=False):
        """Use Phi-3-mini to extract structured action items from transcript"""
        try:
            if is_single_speaker:
                system_prompt = """You are an expert meeting analyst specializing in single-speaker presentations and briefings.

IMPORTANT RULES FOR SINGLE-SPEAKER CONTENT:
- The speaker is presenting information, not assigning tasks to others
- Unless the speaker explicitly says "I will..." or "We will..." or assigns a task to a named person, there are NO actionable tasks
- Do NOT invent people, names, or assignments
- Do NOT assume the speaker is assigning tasks just because they mention future actions

If no explicit action items, commitments, or task assignments are clearly stated, return EXACTLY:
• No actionable tasks identified — this is a presentation/briefing with no assignments

Only extract tasks if the speaker explicitly:
1. Commits to a specific action ("I will prepare the report by Friday")
2. Assigns a task to a named person ("Sarah, please review the proposal")
3. Makes a clear promise or commitment with a deadline

Format each task as:
• WHO: WHAT (deadline if mentioned)
"""
            else:
                system_prompt = """You are an expert meeting analyst specializing in extracting actionable tasks from multi-participant meeting transcripts.

Your expertise includes:
- Identifying concrete, specific action items with clear ownership
- Extracting deadlines, timeframes, and follow-up requirements
- Recognizing commitments, assignments, and next steps
- Distinguishing between decisions and actionable tasks
- Capturing both explicit and implicit task assignments

IMPORTANT: Only extract tasks that are explicitly stated in the transcript. Do not invent tasks, people, or deadlines.

Format each task as:
• WHO: WHAT (deadline if mentioned)
"""
            
            if LANGCHAIN_AVAILABLE and hasattr(self, 'task_extraction_template'):
                messages = self.task_extraction_template.format_messages(
                    meeting_text=text[:4000]
                )
                api_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this meeting transcript and extract actionable tasks:\n\n{text[:4000]}\n\nEXTRACTED TASKS (if any):"}
                ]
            else:
                api_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this meeting transcript and extract actionable tasks:\n\n{text[:4000]}\n\nEXTRACTED TASKS (if any):"}
                ]

            logger.info("⚡ Extracting action items via Phi-3-mini...")
            response = self.client.chat_completion(
                messages=api_messages,
                model=self.LLM_MODEL,
                max_tokens=self._llm_max_tokens,
                temperature=0.2
            )
            choices = response.get("choices", [])
            if not choices or not choices[0].get("message", {}).get("content"):
                logger.warning("LLM returned no choices — falling back to regex")
                return self._extract_tasks_fallback(text)
            content = choices[0]["message"]["content"]
            logger.info("✅ LLM task extraction completed")
            return content.strip()

        except Exception as e:
            logger.warning(f"LLM task extraction failed, falling back to regex: {e}")
            return self._extract_tasks_fallback(text)

    def _extract_tasks_fallback(self, text):
        """Regex-based task extraction fallback when API is unavailable"""
        try:
            task_patterns = [
                r'(?:action item|task|todo|assignment)\s*[:]?\s*([^.!?]{8,120})',
                r'(?:need to|should|must|have to|required to)\s+([^.!?]{8,120})',
                r'([A-Z]\w+(?:\s+[A-Z]\w+)?)\s+(?:will|should|needs to)\s+([^.!?]{8,120})',
                r'(?:by|before|due)\s+\w+day\s*[:]?\s*([^.!?]{8,120})',
                r'(?:complete|finish|send|create|prepare|review|schedule|plan|update|check)\s+([^.!?]{8,120})',
                r'(?:follow up|reach out|contact|coordinate)\s+([^.!?]{8,120})',
            ]
            tasks = []
            processed_text = text[:3000]
            for pattern in task_patterns:
                matches = re.finditer(pattern, processed_text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2 and groups[0] and groups[1]:
                        person = groups[0].strip()
                        action = groups[1].strip()
                        task = f"• {person}: {action}"
                    elif len(groups) >= 1 and groups[0]:
                        action = re.sub(r'^(to\s+|the\s+)', '', groups[0].strip(), flags=re.IGNORECASE)
                        task = f"• {action.capitalize()}"
                    else:
                        continue
                    if len(task) > 10 and len(task) < 200 and task not in tasks:
                        tasks.append(task)
                    if len(tasks) >= 15:
                        break
                if len(tasks) >= 15:
                    break
            if not tasks:
                return "• No specific action items identified in the transcript"
            return "\n".join(tasks)
        except Exception as e:
            logger.error(f"Error in fallback task extraction: {e}")
            return "• Error extracting tasks from meeting content"
    
    def summarize_text(self, text):
        """Summarize via LLM with fallback chain"""
        # Chain: Phi-3-mini (semantic) → BART (API) → keyword fallback
        if self.client and len(text) > 50:
            try:
                return self._llm_executive_summary(text)
            except Exception as e:
                logger.warning(f"LLM summary failed: {e}")

        if self.client and len(text) > 100:
            try:
                return self._bart_summary(text)
            except Exception as e:
                logger.warning(f"BART summary failed: {e}")

        return self.concise_executive_summary(text)

    def _llm_executive_summary(self, text):
        """Use Phi-3-mini to generate semantic executive summary"""
        api_messages = [
            {"role": "system", "content": """You are an expert executive assistant writing meeting minutes.

TASK: Analyze the transcript and write a 2-3 sentence executive summary.

RULES:
- Capture the MOST IMPORTANT points: financial metrics, risk indicators, strategic announcements, outlook
- Be SPECIFIC with numbers (revenue figures, percentages, ratios)
- Do NOT include filler phrases like "the speaker mentioned" or "in this meeting"
- Output ONLY the summary sentences, nothing else
- Do NOT add bullet points or formatting — just plain sentences

EXAMPLE INPUT:
"Our Q1 revenue was $50 million, up 15% year-over-year. We launched the new AI platform last month. Customer retention improved to 92%. We expect continued growth in Q2."

EXAMPLE OUTPUT:
"Q1 revenue reached $50 million, a 15% year-over-year increase, with customer retention improving to 92% following the launch of the new AI platform. Continued growth is expected in Q2."
"""},
            {"role": "user", "content": f"Write a 2-3 sentence executive summary of this transcript:\n\n{text[:3000]}"}
        ]

        response = self.client.chat_completion(
            messages=api_messages,
            model=self.LLM_MODEL,
            max_tokens=200,
            temperature=0.1
        )
        choices = response.get("choices", [])
        if choices and choices[0].get("message", {}).get("content"):
            return choices[0]["message"]["content"].strip()
        raise Exception("No content returned")

    def _bart_summary(self, text):
        """Use BART-SAMSum for summarization"""
        result = self.client.summarization(
            text,
            model=self.SUM_MODEL,
            parameters={"max_length": 150, "min_length": 30, "do_sample": False}
        )
        if isinstance(result, dict):
            return result.get("summary_text", "")
        elif isinstance(result, str):
            return result
        raise Exception("Invalid result format")
    
    def concise_executive_summary(self, text):
        """Produce 2-3 sentence executive summary for minute-taking"""
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if not sentences:
                return text[:150] + "..." if len(text) > 150 else text
            
            # Score sentences by importance
            scored = []
            for i, s in enumerate(sentences):
                score = 0
                lower = s.lower()
                
                # Position: early sentences often contain key info
                if i == 0: score += 3
                if i == 1: score += 2
                if i >= len(sentences) - 2: score += 1
                
                # Financial metrics (highest priority)
                if re.search(r'\$\d|revenue|profit|earnings|financial|forecast|budget', lower):
                    score += 4
                if re.search(r'\d+%|\d+\.\d+%|percent|ratio', lower):
                    score += 3
                
                # Risk & management
                if any(w in lower for w in ['risk', 'loss', 'leverage', 'confidence', 'maximum', 'exceed']):
                    score += 3
                if any(w in lower for w in ['conservative', 'approach', 'manage', 'healthy', 'adopted']):
                    score += 2
                
                # Key business events
                if any(w in lower for w in ['announce', 'launch', 'ipo', 'acquisition', 'merger', 'partnership']):
                    score += 5
                
                # Strategy & outlook
                if any(w in lower for w in ['growth', 'strategy', 'plan', 'target', 'outlook', 'forecast']):
                    score += 3
                
                scored.append((score, i, s))
            
            # Take top 3 sentences, preserve order
            top = sorted(scored, key=lambda x: -x[0])[:3]
            top.sort(key=lambda x: x[1])
            
            summary = ". ".join(s[2] for s in top)
            if not summary.endswith('.'):
                summary += "."
            
            return summary
        except Exception as e:
            logger.error(f"Error in concise summary: {str(e)}")
            return "Summary not available."
    
    def fallback_summary(self, text):
        """Maintain original method for compatibility"""
        return self.concise_executive_summary(text)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment via HF Inference API (emotion-english-distilroberta) on full transcript"""
        try:
            if not self.client:
                return self.fallback_sentiment_analysis(text)

            if len(text) < 20:
                return self.fallback_sentiment_analysis(text)

            logger.info("⚡ Analyzing emotions via HF Inference API (full transcript)...")
            result = self.client.text_classification(
                text,
                model=self.SENT_MODEL
            )
            if isinstance(result, list) and len(result) > 0:
                emotions = result
                top_emotion = max(emotions, key=lambda x: x.get("score", 0))
                label = top_emotion.get("label", "neutral").lower()
                score = top_emotion.get("score", 0)
                
                all_emotions = ", ".join(
                    f"{e.get('label', '?').title()}: {e.get('score', 0)*100:.1f}%" 
                    for e in sorted(emotions, key=lambda x: x.get('score', 0), reverse=True)[:3]
                )
                return f"Primary Emotion: {label.title()} ({score*100:.1f}%) | Top Emotions: {all_emotions}"
            else:
                return self.fallback_sentiment_analysis(text)

        except Exception as e:
            logger.warning(f"API emotion analysis failed, falling back: {e}")
            return self.fallback_sentiment_analysis(text)
    
    def fallback_sentiment_analysis(self, text):
        """Simple sentiment analysis without AI models"""
        try:
            text_lower = text.lower()
            
            positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'agree', 'happy', 'pleased', 'satisfied']
            negative_words = ['bad', 'terrible', 'negative', 'problem', 'issue', 'concern', 'worried', 'disagree', 'failed']
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return "Overall Sentiment: Positive (Based on keyword analysis)"
            elif negative_count > positive_count:
                return "Overall Sentiment: Negative (Based on keyword analysis)"
            else:
                return "Overall Sentiment: Neutral (Based on keyword analysis)"
        
        except Exception:
            return "Sentiment: Neutral"
    
    def identify_key_topics(self, text):
        """Extract semantic key topics as noun phrases"""
        try:
            # Define known business/financial noun phrases to look for
            topic_patterns = [
                (r'\b(ipo|initial public offering)\b', 'IPO'),
                (r'\b(pay\s*plus|payplus)\b', 'PayPlus'),
                (r'\b(revenue|revenues)\b', 'Revenue'),
                (r'\b(blockchain\s+solutions?)\b', 'Blockchain Solutions'),
                (r'\b(predictive\s+analytics?)\b', 'Predictive Analytics'),
                (r'\b(growth\s+strategies?)\b', 'Growth Strategy'),
                (r'\b(risk\s+management?)\b', 'Risk Management'),
                (r'\b(capital\s+ratio)\b', 'Capital Ratio'),
                (r'\b(leverage)\b', 'Leverage'),
                (r'\b(liquidity)\b', 'Liquidity'),
                (r'\b(shareholders?)\b', 'Shareholders'),
                (r'\b(trading\s+day)\b', 'Trading'),
                (r'\b(forecast)\b', 'Forecast'),
                (r'\b(fintech|fin\s*tech)\b', 'FinTech'),
                (r'\b(growth)\b', 'Growth'),
                (r'\b(ai|artificial\s+intelligence)\b', 'AI'),
            ]
            
            topics = []
            text_lower = text.lower()
            
            for pattern, label in topic_patterns:
                if re.search(pattern, text_lower):
                    if label not in topics:
                        topics.append(label)
                    if len(topics) >= 6:
                        break
            
            # Fallback: if no topics found, extract most common content words
            if not topics:
                stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was',
                              'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'that', 'with', 'have', 'this',
                              'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time',
                              'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such',
                              'take', 'than', 'them', 'well', 'were', 'what', 'would', 'there', 'could', 'other',
                              'after', 'first', 'never', 'these', 'think', 'where', 'being', 'thank', 'forward',
                              'also', 'look', 'success', 'faith', 'healthy', 'conservative', 'approach'}
                words = re.findall(r'\b[a-zA-Z]{5,}\b', text_lower)
                word_count = {}
                for w in words:
                    if w not in stop_words:
                        word_count[w] = word_count.get(w, 0) + 1
                top = sorted(word_count.items(), key=lambda x: -x[1])[:5]
                topics = [w.capitalize() for w, c in top]
            
            return "\n".join(f"• {t}" for t in topics[:6])
        
        except Exception as e:
            logger.error(f"Error identifying topics: {str(e)}")
            return "• Error analyzing topics"
    
    def process_meeting_simple(self, audio_file, progress=None):
        """Process meeting audio via HF Inference API with chunked processing for long audio"""
        if audio_file is None:
            return "", None
        
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting meeting processing via HF Inference API...")
            
            if not self.client:
                return "HF_TOKEN not configured. Set your Hugging Face read token in the Space settings.", None
            
            # Get audio duration
            duration = self._get_audio_duration(audio_file)
            is_long_audio = duration > self.CHUNK_DURATION * 2
            
            if is_long_audio:
                logger.info(f"⚡ Long audio detected ({duration:.1f}s), using chunked processing")
            
            # Step 1: Transcription
            if progress is not None:
                progress(0.1, desc="Transcribing audio...")
            
            if is_long_audio:
                transcription = self._chunked_transcription(audio_file, progress)
            else:
                result = self.client.automatic_speech_recognition(
                    audio_file,
                    model=self.ASR_MODEL
                )
                logger.info(f"  ASR result type: {type(result).__name__}")
                if isinstance(result, dict):
                    transcription = result.get("text", "")
                elif isinstance(result, str):
                    transcription = result
                else:
                    logger.warning(f"  Unexpected ASR result: {result}")
                    transcription = ""
                logger.info(f"  Transcription length: {len(transcription)} chars")
            
            if not transcription:
                return "No speech detected in the audio file.", None
            
            transcript_time = time.time() - start_time
            logger.info(f"⚡ Transcription completed in {transcript_time:.2f}s")
            
            # Step 1.5: Speaker count (default — no diarization)
            speaker_count = 1
            is_single_speaker = True
            
            # Step 2: Analysis
            if progress is not None:
                progress(0.3, desc="Running parallel analysis...")
            
            analysis_start = time.time()
            
            # Use map-reduce for long transcripts, parallel for short
            if is_long_audio and len(transcription) > 3000:
                # Long transcript - use map-reduce summarization
                summary = self._map_reduce_summarize(transcription, progress)
                sentiment = self._segmented_sentiment(transcription, progress)
                
                # Run action items and topics in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_actions = executor.submit(self.extract_action_items, transcription, is_single_speaker)
                    fut_topics = executor.submit(self.identify_key_topics, transcription)
                    action_items = fut_actions.result()
                    key_topics = fut_topics.result()
            else:
                # Short transcript - run all in parallel
                logger.info(f"⚡ Running parallel analysis on {len(transcription)} chars...")
                with ThreadPoolExecutor(max_workers=4) as executor:
                    fut_summary = executor.submit(self.summarize_text, transcription)
                    fut_actions = executor.submit(self.extract_action_items, transcription, is_single_speaker)
                    fut_sentiment = executor.submit(self.analyze_sentiment, transcription)
                    fut_topics = executor.submit(self.identify_key_topics, transcription)
                    
                    summary = fut_summary.result()
                    logger.info("  ✅ Summary done")
                    action_items = fut_actions.result()
                    logger.info("  ✅ Action items done")
                    sentiment = fut_sentiment.result()
                    logger.info("  ✅ Sentiment done")
                    key_topics = fut_topics.result()
                    logger.info("  ✅ Topics done")
            
            analysis_time = time.time() - analysis_start
            logger.info(f"⚡ Analysis completed in {analysis_time:.2f}s")
            
            if progress is not None:
                progress(0.85, desc="Generating report...")
            
            # Step 3: Generate comprehensive report
            meeting_minutes = self._generate_meeting_report(
                transcription, summary, action_items, sentiment, key_topics, duration, speaker_count
            )
            
            total_time = time.time() - start_time
            
            # Create download file
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', prefix='meeting_minutes_')
            temp_file.write(meeting_minutes)
            temp_file.close()
            
            logger.info(f"✅ Processing completed in {total_time:.2f}s (serverless)")
            return meeting_minutes, temp_file.name
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing meeting: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_msg, None
    
    def _get_audio_duration(self, audio_path):
        """Get audio duration in seconds using wave module"""
        try:
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 0
    
    def _compress_audio(self, audio_path, output_path=None):
        """Compress audio to 16kHz mono WAV for faster processing"""
        try:
            import torch
            import torchaudio
            
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.wav')
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Normalize audio
            waveform = waveform / (waveform.abs().max() + 1e-8)
            
            torchaudio.save(output_path, waveform, sample_rate)
            logger.info(f"✅ Audio compressed to {output_path}")
            return output_path
        except ImportError:
            logger.warning("⚠️ torch/torchaudio not available, skipping compression")
            return audio_path
        except Exception as e:
            logger.warning(f"⚠️ Audio compression failed: {e}")
            return audio_path
    
    def _create_audio_chunks(self, audio_path, chunk_duration=None, overlap=None):
        """Split audio into overlapping chunks for processing"""
        try:
            import torch
            import torchaudio
            
            if chunk_duration is None:
                chunk_duration = self.CHUNK_DURATION
            if overlap is None:
                overlap = self.CHUNK_OVERLAP
            
            waveform, sample_rate = torchaudio.load(audio_path)
            total_samples = waveform.shape[1]
            chunk_samples = int(chunk_duration * sample_rate)
            overlap_samples = int(overlap * sample_rate)
            step_samples = chunk_samples - overlap_samples
            
            chunks = []
            chunk_dir = tempfile.mkdtemp(prefix='meeting_chunks_')
            
            start = 0
            chunk_idx = 0
            
            while start < total_samples:
                end = min(start + chunk_samples, total_samples)
                chunk_waveform = waveform[:, start:end]
                
                chunk_path = os.path.join(chunk_dir, f'chunk_{chunk_idx:04d}.wav')
                torchaudio.save(chunk_path, chunk_waveform, sample_rate)
                
                chunks.append({
                    'path': chunk_path,
                    'start_time': start / sample_rate,
                    'end_time': end / sample_rate,
                    'index': chunk_idx
                })
                
                chunk_idx += 1
                start += step_samples
                
                # Stop if we've reached the end
                if end >= total_samples:
                    break
            
            logger.info(f"✅ Created {len(chunks)} audio chunks")
            return chunks, chunk_dir
        except ImportError:
            logger.warning("⚠️ torch/torchaudio not available for chunking")
            return [{'path': audio_path, 'start_time': 0, 'end_time': 0, 'index': 0}], None
        except Exception as e:
            logger.warning(f"⚠️ Audio chunking failed: {e}")
            return [{'path': audio_path, 'start_time': 0, 'end_time': 0, 'index': 0}], None
    
    def _chunked_transcription(self, audio_path, progress=None):
        """Transcribe audio in chunks for long meetings"""
        try:
            # Check if audio needs chunking
            duration = self._get_audio_duration(audio_path)
            logger.info(f"📊 Audio duration: {duration:.1f}s")
            
            if duration <= self.CHUNK_DURATION * 2:
                # Short audio - transcribe directly
                if progress is not None:
                    progress(0.15, desc="Transcribing audio...")
                result = self.client.automatic_speech_recognition(
                    audio_path,
                    model=self.ASR_MODEL
                )
                return result.get("text", "")
            
            # Long audio - chunk and transcribe
            logger.info(f"⚡ Long audio detected ({duration:.1f}s), using chunked transcription")
            chunks, chunk_dir = self._create_audio_chunks(audio_path)
            
            transcriptions = []
            for i, chunk in enumerate(chunks):
                if progress is not None:
                    progress_val = 0.1 + (0.2 * (i / len(chunks)))
                    progress(progress_val, desc=f"Transcribing chunk {i+1}/{len(chunks)}...")
                
                logger.info(f"⚡ Transcribing chunk {i+1}/{len(chunks)} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s)")
                
                result = self.client.automatic_speech_recognition(
                    chunk['path'],
                    model=self.ASR_MODEL
                )
                if isinstance(result, dict):
                    chunk_text = result.get("text", "")
                elif isinstance(result, str):
                    chunk_text = result
                else:
                    chunk_text = ""
                if chunk_text:
                    transcriptions.append(chunk_text)
            
            # Clean up chunk directory
            if chunk_dir and os.path.exists(chunk_dir):
                import shutil
                shutil.rmtree(chunk_dir, ignore_errors=True)
            
            # Combine transcriptions with overlap handling
            full_transcript = " ".join(transcriptions)
            
            # Remove duplicate sentences from overlap
            sentences = re.split(r'(?<=[.!?])\s+', full_transcript)
            unique_sentences = []
            seen = set()
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen:
                    seen.add(sentence)
                    unique_sentences.append(sentence)
            
            return " ".join(unique_sentences)
        except Exception as e:
            logger.error(f"Error in chunked transcription: {e}")
            raise
    
    def _map_reduce_summarize(self, text, progress=None):
        """Map-reduce summarization for long transcripts"""
        try:
            # Split text into chunks for summarization
            max_chunk_size = 1000  # tokens approx
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence_tokens = len(sentence.split())
                if current_size + sentence_tokens > max_chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_tokens
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            if len(chunks) <= 1:
                # Short text - summarize directly
                return self.summarize_text(text)
            
            # Map phase: summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                if progress is not None:
                    progress_val = 0.3 + (0.4 * (i / len(chunks)))
                    progress(progress_val, desc=f"Summarizing chunk {i+1}/{len(chunks)}...")
                
                summary = self.summarize_text(chunk)
                chunk_summaries.append(summary)
            
            # Reduce phase: combine summaries
            combined_summaries = "\n".join(chunk_summaries)
            
            # Final summary
            if progress is not None:
                progress(0.7, desc="Generating final summary...")
            
            final_summary = self.summarize_text(combined_summaries)
            return final_summary
        except Exception as e:
            logger.warning(f"Map-reduce summarization failed: {e}")
            return self.concise_executive_summary(text)
    
    def _segmented_sentiment(self, text, progress=None):
        """Sentiment analysis segmented by speaker or time"""
        try:
            # Split text into segments (by paragraphs or sentences)
            segments = re.split(r'\n\n+', text)
            if len(segments) <= 1:
                segments = re.split(r'(?<=[.!?])\s+', text)
            
            # Analyze each segment
            segment_sentiments = []
            for i, segment in enumerate(segments):
                if len(segment.strip()) < 20:
                    continue
                
                if progress and i % 5 == 0:
                    progress_val = 0.7 + (0.15 * (i / len(segments)))
                    progress(progress_val, desc=f"Analyzing sentiment {i+1}/{len(segments)}...")
                
                sentiment = self.analyze_sentiment(segment)
                segment_sentiments.append({
                    'segment': segment[:100] + "..." if len(segment) > 100 else segment,
                    'sentiment': sentiment
                })
            
            # Aggregate sentiments
            if not segment_sentiments:
                return self.analyze_sentiment(text)
            
            # Count emotion occurrences
            emotion_counts = {}
            for seg in segment_sentiments:
                # Extract emotion from sentiment string
                emotion_match = re.search(r'Primary Emotion: (\w+)', seg['sentiment'])
                if emotion_match:
                    emotion = emotion_match.group(1)
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Build aggregated result
            if emotion_counts:
                total = sum(emotion_counts.values())
                top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                emotion_str = ", ".join([f"{e}: {c/total*100:.1f}%" for e, c in top_emotions])
                return f"Overall Sentiment: {emotion_str} (Based on {len(segment_sentiments)} segments)"
            else:
                return segment_sentiments[0]['sentiment'] if segment_sentiments else self.analyze_sentiment(text)
        except Exception as e:
            logger.warning(f"Segmented sentiment failed: {e}")
            return self.analyze_sentiment(text)
    
    def _save_checkpoint(self, job_id, data):
        """Save processing checkpoint"""
        try:
            os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(self.CHECKPOINT_DIR, f"{job_id}.json")
            with open(checkpoint_path, 'w') as f:
                json.dump(data, f)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save checkpoint: {e}")
    
    def _load_checkpoint(self, job_id):
        """Load processing checkpoint"""
        try:
            checkpoint_path = os.path.join(self.CHECKPOINT_DIR, f"{job_id}.json")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
                return data
        except Exception as e:
            logger.warning(f"⚠️ Could not load checkpoint: {e}")
        return None
    
    def _generate_meeting_report(self, transcript, summary, actions, sentiment, topics, duration=None, speaker_count=1):
        """Generate meeting report with brutalist terminal formatting"""
        sep = "=" * 52
        
        # Duration string
        duration_str = f"\n>> DURATION\n{duration:.1f} seconds ({duration/60:.1f} minutes)" if duration else ""
        
        # Meeting type indicator
        if speaker_count <= 1:
            meeting_type = "PRESENTATION / BRIEFING"
        elif speaker_count == 2:
            meeting_type = "1-ON-1 MEETING"
        else:
            meeting_type = f"MULTI-PARTICIPANT MEETING ({speaker_count} speakers)"
        
        # Sentiment header
        sentiment_header = "SENTIMENT"
        
        # Task list with meeting-type awareness
        if speaker_count <= 1:
            # Single speaker - check if actions indicate no tasks
            if "No actionable tasks" in actions or "no assignments" in actions.lower():
                task_section = f">> ACTION ITEMS\n{actions}"
            else:
                task_section = f">> ACTION ITEMS\n{actions}"
        else:
            task_section = f">> ACTION ITEMS\n{actions}"
        
        return f"""{sep}
                MEETING MINUTES
{sep}

>> MEETING TYPE
{meeting_type}
{duration_str}

>> EXECUTIVE SUMMARY
{summary}

{task_section}

>> {sentiment_header}
{sentiment}

>> KEY DISCUSSION POINTS
{topics}

{sep}
        GENERATED BY AI MEETING ASSISTANT
{sep}"""


def process_meeting_audio(audio_file, progress=gr.Progress()):
    if audio_file is None:
        return "Please upload an audio file to analyze.", None
    
    # Validate audio duration
    try:
        with wave.open(audio_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            
            if duration > meeting_assistant.MAX_AUDIO_DURATION:
                return f"Error: Audio file too long ({duration:.1f}s). Maximum allowed duration is {meeting_assistant.MAX_AUDIO_DURATION/60:.0f} minutes.", None
            
            if duration < 1.0:
                return "Error: Audio file too short. Please upload a longer audio file.", None
            
            logger.info(f"📊 Audio file duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    except Exception as e:
        logger.warning(f"⚠️ Could not validate audio duration: {e}")

    meeting_report, temp_file = meeting_assistant.process_meeting_simple(audio_file, progress=progress)

    if meeting_report and not meeting_report.startswith("Error"):
        return meeting_report, temp_file
    else:
        return "Error processing audio file. Please try again.", None


def clear_interface():
    return None, "", None



# Gradio 6.x – custom JS for brutalist theme, animations, and Gradio footer removal
custom_js = """
function() {
    document.documentElement.classList.add('brutalist');
    document.body.classList.add('brutalist');
    
    if (window.matchMedia) {
        var origMatchMedia = window.matchMedia;
        window.matchMedia = function(query) {
            if (query.includes('prefers-color-scheme')) {
                return {
                    matches: false,
                    media: query,
                    addEventListener: function() {},
                    removeEventListener: function() {},
                    addListener: function() {},
                    removeListener: function() {}
                };
            }
            return origMatchMedia(query);
        };
    }
    
    setTimeout(function() {
        document.body.classList.add('loaded');
        var staggerEls = document.querySelectorAll('.stagger-in');
        staggerEls.forEach(function(el, i) {
            setTimeout(function() {
                el.classList.add('visible');
            }, i * 100);
        });
    }, 300);
    
    var hideFooter = setInterval(function() {
        var footer = document.querySelector('.built-with-gradio, gradio-app footer:not(.brutal-footer)');
        if (footer) {
            footer.style.display = 'none';
            footer.remove();
            clearInterval(hideFooter);
        }
    }, 150);
    setTimeout(function() { clearInterval(hideFooter); }, 8000);
    
    setTimeout(function() {
        var outputWrapper = document.querySelector('#output-terminal');
        if (outputWrapper) {
            var observer = new MutationObserver(function() {
                var textarea = outputWrapper.querySelector('textarea');
                if (textarea && textarea.value && textarea.value.trim().length > 0) {
                    textarea.classList.add('has-content');
                }
            });
            observer.observe(outputWrapper, { childList: true, subtree: true, characterData: true });
        }
    }, 1000);
    
    /* ═══════════════════════════════════════════
       UPLOAD STATUS — global helpers + robust listeners
       ═══════════════════════════════════════════ */
    window.__uploadStatus = {
        el: null,
        _get: function() {
            if (!this.el) this.el = document.getElementById('upload-status');
            return this.el;
        },
        set: function(state, text) {
            var el = this._get();
            if (!el) return;
            el.className = 'upload-status ' + (state || '');
            var txt = el.querySelector('.status-text');
            if (txt) txt.textContent = text || 'AWAITING INPUT';
        }
    };

    /* Detect native file input change (manual upload) */
    document.addEventListener('change', function(e) {
        var t = e.target;
        if (t && t.tagName === 'INPUT' && t.type === 'file' && t.closest && t.closest('#audio-upload') && t.files && t.files.length > 0) {
            window.__uploadStatus.set('uploading', 'UPLOADING...');
        }
    }, true);

    /* MutationObserver — catch sample clicks, uploads, recording */
    var __audioObs = setInterval(function() {
        var uploadZone = document.getElementById('audio-upload');
        if (!uploadZone) return;
        clearInterval(__audioObs);

        var obs = new MutationObserver(function() {
            var audio = uploadZone.querySelector('audio');
            if (audio && audio.src && audio.src !== window.location.href) {
                // Audio element appeared — uploading or loading sample
                window.__uploadStatus.set('uploading', 'LOADING AUDIO...');
                if (audio.addEventListener) {
                    audio.addEventListener('canplay', function() {
                        window.__uploadStatus.set('complete', 'AUDIO READY');
                    }, { once: true });
                    audio.addEventListener('error', function() {
                        window.__uploadStatus.set('', 'AWAITING INPUT');
                    }, { once: true });
                }
            }
        });
        obs.observe(uploadZone, { childList: true, subtree: true });
    }, 200);

    /* CLEAR button — reset status */
    document.addEventListener('click', function(e) {
        if (e.target && e.target.closest && e.target.closest('#btn-clear')) {
            setTimeout(function() { window.__uploadStatus.set('', 'AWAITING INPUT'); }, 100);
        }
    });

    /* Watch output terminal for processing completion */
    var __termReady = setInterval(function() {
        var term = document.getElementById('output-terminal');
        if (!term) return;
        clearInterval(__termReady);
        var tobs = new MutationObserver(function() {
            var ta = term.querySelector('textarea');
            if (ta && ta.value && ta.value.trim().length > 0
                && ta.value.indexOf('Error') !== 0
                && ta.value.indexOf('Please upload') !== 0) {
                window.__uploadStatus.set('complete', 'PROCESSING COMPLETE');
                var indicator = document.getElementById('processing-indicator');
                if (indicator) {
                    indicator.classList.remove('active');
                }
                setTimeout(function() { window.__uploadStatus.set('', 'AWAITING INPUT'); }, 3500);
            }
        });
        tobs.observe(term, { childList: true, subtree: true, characterData: true });
    }, 300);
    
    return [];
}
"""


def create_interface():
    """Create brutalist-industrial Gradio interface"""
    
    with gr.Blocks(
        title="AI Meeting Assistant"
    ) as interface:
        
        # ═══════════════════════════════════════════
        # BRUTALIST INDUSTRIAL CSS SYSTEM
        # ═══════════════════════════════════════════
        gr.HTML("""
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;1,400&display=swap" rel="stylesheet">
<style>
/* ═══ CSS VARIABLES ═══ */
:root {
    --bg: #0a0a0a;
    --bg2: #111111;
    --bg3: #1a1a1a;
    --txt: #e8e6e3;
    --txt2: #999999;
    --txt3: #555555;
    --acc: #ff6b35;
    --acc2: #ff8c5a;
    --acc-g: rgba(255,107,53,0.10);
    --bdr: #2a2a2a;
    --bdr2: #3a3a3a;
    --fh: 'Bebas Neue', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --t: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ═══ BASE RESET ═══ */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{background:var(--bg);scroll-behavior:smooth;height:100%}
body{
    background:var(--bg);
    color:var(--txt);
    font-family:var(--fm);
    font-weight:300;
    font-size:13px;
    line-height:1.7;
    -webkit-font-smoothing:antialiased;
    overflow-x:hidden;
    min-height:100vh;
}
::selection{background:var(--acc);color:var(--bg)}

/* ═══ ATMOSPHERE: GRID LINES ═══ */
body::before{
    content:'';
    position:fixed;inset:0;
    pointer-events:none;z-index:-1;
    background:
        linear-gradient(180deg,var(--bg) 0%,rgba(10,10,10,0.97) 30%,rgba(10,10,10,0.97) 70%,var(--bg) 100%),
        linear-gradient(rgba(255,107,53,0.012) 1px,transparent 1px),
        linear-gradient(90deg,rgba(255,107,53,0.012) 1px,transparent 1px);
    background-size:100% 100%,64px 64px,64px 64px;
}

/* ═══ SCROLLBAR ═══ */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg2)}
::-webkit-scrollbar-thumb{background:var(--bdr2)}
::-webkit-scrollbar-thumb:hover{background:var(--acc)}

/* ═══ GRADIO CONTAINER OVERRIDES ═══ */
.gradio-container{max-width:100%!important;width:100%!important;padding:0!important;background:transparent!important;margin:0!important}
.gradio-container .contain{max-width:100%!important;min-width:100%!important;width:100%!important;margin:0 auto!important;padding:0 3rem!important}
.gradio-container .app{background:transparent!important}
.gradio-container .main{background:transparent!important}
.gr-box,.gr-panel,.gr-form{background:transparent!important;border:none!important;box-shadow:none!important;border-radius:0!important}
.prose{color:var(--txt)!important}
.prose *{color:var(--txt)!important}

/* Hide Gradio branding only (not our custom footer) */
gradio-app footer:not(.brutal-footer),#footer,.built-with-gradio{display:none!important;visibility:hidden!important;height:0!important;overflow:hidden!important;opacity:0!important}

/* ═══ GRADIO LABEL OVERRIDES ═══ */
.gr-html .brutal-header,
div.brutal-header,
.brutal-header *{
    text-align:center!important;
}
label,.gr-input-label,.gr-audio label,.gr-textbox label,.gr-file label{
    font-family:var(--fh)!important;
    font-weight:400!important;
    text-transform:uppercase!important;
    letter-spacing:0.12em!important;
    color:var(--txt2)!important;
    font-size:0.7rem!important;
    margin-bottom:0.5rem!important;
}

/* ═══ HERO HEADER ═══ */
.brutal-header{
    text-align:center!important;
    padding:3.5rem 2.5rem 1.5rem;
    max-width:100%;
    margin:0 auto;
    position:relative;
}
.header-rule{
    width:100%;
    height:1px;
    margin:1.5rem 0;
    background:linear-gradient(90deg,transparent,var(--bdr2) 15%,var(--bdr2) 85%,transparent);
}
.header-title{
    font-family:var(--fh);
    font-size:clamp(3.5rem,8.5vw,7rem);
    color:var(--txt);
    line-height:0.82;
    letter-spacing:-0.03em;
    text-transform:uppercase;
    margin:0;
    text-shadow:0 0 80px rgba(255,107,53,0.05);
}
.header-subtitle{
    font-family:var(--fh);
    font-size:clamp(0.75rem,1.4vw,0.9rem);
    color:var(--acc);
    letter-spacing:0.3em;
    text-transform:uppercase;
    display:flex;
    align-items:center;
    justify-content:center;
    gap:1rem;
    margin:0.5rem 0;
}
.subtitle-dot{
    display:inline-block;
    width:5px;height:5px;
    background:var(--acc);
    animation:dotPulse 2s ease-in-out infinite;
}
.header-desc{
    font-family:var(--fm);
    font-size:0.75rem;
    font-weight:300;
    color:var(--txt3);
    letter-spacing:0.05em;
    line-height:1.8;
    max-width:600px;
    margin-left:auto!important;
    margin-right:auto!important;
    text-align:center!important;
}
.header-desc strong{color:var(--txt2);font-weight:500}

/* ═══ MAIN CONTENT GRID ═══ */
.main-content{
    display:flex!important;
    gap:0!important;
    max-width:100%!important;
    margin:0 auto!important;
    padding:0 3rem!important;
    flex-wrap:wrap!important;
}
.main-content > .gr-column{padding:0!important}
.content-panel{
    padding:1.5rem!important;
    position:relative!important;
    border:1px solid var(--bdr)!important;
    background:var(--bg2)!important;
    min-height:420px!important;
}
.upload-panel{
    flex:11!important;
    min-width:300px!important;
    border-right:none!important;
}
.output-panel{
    flex:9!important;
    min-width:280px!important;
    display:flex!important;
    flex-direction:column!important;
}

/* ═══ PANEL LABELS ═══ */
.panel-label{
    font-family:var(--fh);
    font-size:0.8rem;
    letter-spacing:0.2em;
    color:var(--acc);
    text-transform:uppercase;
    margin-bottom:1.25rem;
    padding-bottom:0.75rem;
    border-bottom:1px solid var(--bdr);
    display:flex;
    align-items:center;
    gap:0.5rem;
}
.panel-label-arrow{
    color:var(--acc);
    font-family:var(--fm);
    font-size:0.7rem;
    animation:arrowBlink 1.5s step-end infinite;
}

/* ═══ AUDIO UPLOAD ZONE ═══ */
#audio-upload{
    border:1px dashed var(--bdr)!important;
    background:var(--bg3)!important;
    padding:1.5rem!important;
    transition:all var(--t)!important;
    min-height:120px!important;
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
}
#audio-upload:hover{
    border-color:var(--acc)!important;
    background:rgba(255,107,53,0.03)!important;
}
#audio-upload audio{width:100%!important;border-radius:0!important}
#audio-upload input[type="file"]{color:var(--txt)!important;font-family:var(--fm)!important}
#audio-upload .audio-container{width:100%!important}

/* Upload / Record source toggles (override global button padding) */
#audio-upload .source-selection{
    display:flex!important;
    gap:0.5rem!important;
    margin-top:0.75rem!important;
    justify-content:center!important;
}
#audio-upload button.icon{
    padding:0.4rem 0.85rem!important;
    min-height:2rem!important;
    height:auto!important;
    width:auto!important;
    display:inline-flex!important;
    align-items:center!important;
    justify-content:center!important;
    gap:0.4rem!important;
    font-family:var(--fm)!important;
    font-size:0.65rem!important;
    font-weight:400!important;
    letter-spacing:0.1em!important;
    text-transform:uppercase!important;
    border-radius:0!important;
    background:var(--bg)!important;
    cursor:pointer!important;
    transition:all var(--t)!important;
    line-height:1!important;
}
#audio-upload button.icon svg{
    width:14px!important;
    height:14px!important;
    flex-shrink:0!important;
}
#audio-upload button.icon.selected{
    color:var(--acc)!important;
    border-color:var(--acc)!important;
    background:rgba(255,107,53,0.08)!important;
}
#audio-upload button.icon.selected svg,
#audio-upload button.icon.selected svg *{stroke:var(--acc)!important}
#audio-upload button.icon:not(.selected){
    color:var(--txt2)!important;
    border-color:var(--bdr2)!important;
}
#audio-upload button.icon:not(.selected) svg,
#audio-upload button.icon:not(.selected) svg *{stroke:var(--txt2)!important}
#audio-upload button.icon:not(.selected):hover{
    color:var(--txt)!important;
    border-color:var(--txt2)!important;
}
#audio-upload button[aria-label="Upload file"]::after{content:"UPLOAD"}
#audio-upload button[aria-label="Record audio"]::after{content:"RECORD"}

/* ═══ UPLOAD STATUS INDICATOR ═══ */
.upload-status{
    display:flex;
    align-items:center;
    gap:0.5rem;
    padding:0.6rem 0.85rem;
    margin-bottom:0.85rem;
    background:var(--bg3);
    border:1px solid var(--bdr);
    font-family:var(--fm);
    font-size:0.65rem;
    letter-spacing:0.1em;
    text-transform:uppercase;
    color:var(--txt2);
    transition:all var(--t);
    position:relative;
    overflow:hidden;
}
.upload-status.uploading{
    border-color:var(--acc);
    color:var(--acc);
    background:rgba(255,107,53,0.06);
    box-shadow:inset 0 0 18px rgba(255,107,53,0.08);
}
.upload-status.uploading::after{
    content:'';
    position:absolute;
    top:0;left:-100%;
    width:60%;
    height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,107,53,0.08),transparent);
    animation:shimmer 1.4s ease-in-out infinite;
}
.upload-status.processing{
    border-color:var(--acc);
    color:var(--acc);
    background:rgba(255,107,53,0.08);
    box-shadow:inset 0 0 22px rgba(255,107,53,0.10);
}
.upload-status.processing::after{
    content:'';
    position:absolute;
    top:0;left:-100%;
    width:40%;
    height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,107,53,0.12),transparent);
    animation:shimmer 0.9s ease-in-out infinite;
}
.upload-status.complete{
    border-color:#22c55e;
    color:#22c55e;
    background:rgba(34,197,94,0.06);
}
.status-dot{
    width:10px;
    height:10px;
    border-radius:50%;
    background:currentColor;
    flex-shrink:0;
    opacity:0.6;
    transition:all var(--t);
    position:relative;
    z-index:1;
}
.status-text{position:relative;z-index:1}
.upload-status.uploading .status-dot,
.upload-status.processing .status-dot{
    opacity:1;
    animation:statusPulse 0.8s ease-in-out infinite;
}
.upload-status.complete .status-dot{
    opacity:1;
    background:#22c55e;
    animation:none;
}
@keyframes statusPulse{
    0%,100%{transform:scale(1);opacity:1;box-shadow:0 0 4px currentColor}
    50%{transform:scale(1.8);opacity:0.4;box-shadow:0 0 12px currentColor}
}
@keyframes shimmer{
    0%{left:-100%}
    100%{left:160%}
}

/* ═══ OUTPUT TERMINAL ═══ */
#output-terminal{flex:1!important;display:flex!important;flex-direction:column!important}
#output-terminal > div{flex:1!important}
#output-terminal textarea{
    width:100%!important;
    min-height:380px!important;
    height:100%!important;
    background:var(--bg)!important;
    color:var(--txt)!important;
    font-family:var(--fm)!important;
    font-size:12px!important;
    font-weight:300!important;
    line-height:1.8!important;
    border:1px solid var(--bdr)!important;
    border-radius:0!important;
    padding:1.25rem!important;
    resize:none!important;
    outline:none!important;
    transition:all var(--t)!important;
    box-shadow:inset 0 0 40px rgba(0,0,0,0.4)!important;
    caret-color:var(--acc)!important;
    background-image:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.04) 2px,rgba(0,0,0,0.04) 4px)!important;
}
#output-terminal textarea:focus{
    border-color:var(--acc)!important;
    box-shadow:inset 0 0 40px rgba(0,0,0,0.4),0 0 0 1px var(--acc-g)!important;
}
#output-terminal textarea::placeholder{color:var(--txt3)!important;font-style:italic}
#output-terminal textarea.has-content{border-color:var(--bdr2)!important}

/* ═══ BUTTONS ═══ */
button,.gr-button{
    font-family:var(--fh)!important;
    font-size:0.9rem!important;
    letter-spacing:0.15em!important;
    text-transform:uppercase!important;
    border-radius:0!important;
    border:1px solid!important;
    padding:0.7rem 1.8rem!important;
    cursor:pointer!important;
    transition:all 0.15s cubic-bezier(0.4,0,0.2,1)!important;
    position:relative!important;
    outline:none!important;
}
#btn-submit{
    background:var(--acc)!important;
    color:var(--bg)!important;
    border-color:var(--acc)!important;
    font-weight:400!important;
}
#btn-submit:hover{
    background:transparent!important;
    color:var(--acc)!important;
    box-shadow:0 0 24px var(--acc-g)!important;
}
#btn-submit:active{transform:scale(0.96)!important;box-shadow:0 0 8px var(--acc-g)!important}
#btn-clear{
    background:transparent!important;
    color:var(--txt2)!important;
    border-color:var(--bdr2)!important;
    font-weight:400!important;
}
#btn-clear:hover{color:var(--txt)!important;border-color:var(--txt2)!important}
#btn-clear:active{transform:scale(0.96)!important}
.button-row{display:flex!important;gap:0.75rem!important;margin-top:1rem!important}
.button-row button{flex:1!important}

/* ═══ EXAMPLES ═══ */
.gr-examples{margin-top:1rem!important}
.gr-examples .examples-title{
    font-family:var(--fh)!important;
    font-size:0.7rem!important;
    letter-spacing:0.12em!important;
    text-transform:uppercase!important;
    color:var(--txt3)!important;
    margin-bottom:0.5rem!important;
}
.gr-examples table{width:100%!important;border-collapse:collapse!important}
.gr-examples td{padding:0!important}
.gr-examples button{
    width:100%!important;
    background:var(--bg3)!important;
    color:var(--txt2)!important;
    border:1px solid var(--bdr)!important;
    font-family:var(--fm)!important;
    font-size:0.7rem!important;
    font-weight:400!important;
    text-transform:none!important;
    letter-spacing:0.03em!important;
    padding:0.6rem 1rem!important;
    text-align:left!important;
}
.gr-examples button:hover{background:var(--bg)!important;color:var(--acc)!important;border-color:var(--acc)!important}

/* ═══ FILE DOWNLOAD ═══ */
#download-file{
    margin-top:0.75rem!important;
    border:1px solid var(--bdr)!important;
    background:var(--bg3)!important;
    padding:0.5rem!important;
}
#download-file .file-preview{font-family:var(--fm)!important;font-size:0.7rem!important;color:var(--txt2)!important}
.download-section{
    font-family:var(--fh);
    font-size:0.7rem;
    letter-spacing:0.15em;
    color:var(--txt3);
    text-transform:uppercase;
    margin-top:1rem;
    padding-top:0.75rem;
    border-top:1px solid var(--bdr);
    display:flex;
    align-items:center;
    gap:0.5rem;
}

/* ═══ FOOTER ═══ */
.brutal-footer{text-align:center;padding:2.5rem 2.5rem 3rem;max-width:100%;margin:0 auto}
.footer-rule{
    width:100%;
    height:1px;
    margin-bottom:1.5rem;
    background:linear-gradient(90deg,transparent,var(--bdr) 30%,var(--bdr) 70%,transparent);
}
.footer-text{
    font-family:var(--fm);
    font-size:0.65rem;
    color:var(--txt3);
    letter-spacing:0.08em;
    text-transform:uppercase;
}
.footer-text span{color:var(--acc)}

/* ═══ PROCESSING STATE ═══ */
#btn-submit:disabled{opacity:0.5!important;cursor:not-allowed!important;animation:pulse 1.5s ease-in-out infinite!important}

/* ═══ LONG AUDIO PROCESSING ═══ */
.processing-indicator{
    display:none;
    align-items:center;
    gap:0.5rem;
    padding:0.6rem 0.85rem;
    margin-bottom:0.85rem;
    background:rgba(255,107,53,0.08);
    border:1px solid var(--acc);
    font-family:var(--fm);
    font-size:0.65rem;
    letter-spacing:0.1em;
    text-transform:uppercase;
    color:var(--acc);
    animation:statusPulse 1s ease-in-out infinite;
}
.processing-indicator.active{
    display:flex;
}
.processing-indicator .progress-bar{
    flex:1;
    height:2px;
    background:var(--bdr);
    position:relative;
    overflow:hidden;
}
.processing-indicator .progress-bar::after{
    content:'';
    position:absolute;
    top:0;left:0;
    width:30%;
    height:100%;
    background:var(--acc);
    animation:progressSlide 1.5s ease-in-out infinite;
}
@keyframes progressSlide{
    0%{left:-30%}
    100%{left:130%}
}

/* ═══ AUDIO INFO ═══ */
.audio-info{
    font-family:var(--fm);
    font-size:0.6rem;
    color:var(--txt3);
    margin-top:0.5rem;
    padding:0.4rem;
    background:var(--bg);
    border:1px solid var(--bdr);
}
.audio-info .duration{
    color:var(--txt2);
}
.audio-info .chunk-info{
    color:var(--acc);
    margin-top:0.2rem;
}

/* ═══ ANIMATIONS ═══ */
@keyframes dotPulse{0%,100%{opacity:0.3;transform:scale(0.8)}50%{opacity:1;transform:scale(1.2)}}
@keyframes arrowBlink{0%,100%{opacity:0.3}50%{opacity:1}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.6}}
@keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes terminalFlash{
    0%{box-shadow:inset 0 0 40px rgba(0,0,0,0.4)}
    50%{box-shadow:inset 0 0 40px rgba(0,0,0,0.4),inset 0 0 2px var(--acc)}
    100%{box-shadow:inset 0 0 40px rgba(0,0,0,0.4)}
}

/* ═══ ENTRANCE ANIMATION (CSS-only; no JS required) ═══ */
.stagger-in{
    opacity:1;
    animation:fadeInUp 0.6s cubic-bezier(0.4,0,0.2,1) both;
    animation-delay:0.15s;
}
#output-terminal textarea.has-content{animation:terminalFlash 0.6s ease-out 1}
body.loaded .header-title{animation:fadeInUp 0.8s cubic-bezier(0.4,0,0.2,1) both;animation-delay:0.1s}
body.loaded .header-subtitle{animation:fadeInUp 0.8s cubic-bezier(0.4,0,0.2,1) both;animation-delay:0.2s}
body.loaded .header-desc{animation:fadeInUp 0.8s cubic-bezier(0.4,0,0.2,1) both;animation-delay:0.3s}

/* ═══ RESPONSIVE ═══ */
@media (max-width:768px){
    .gradio-container .contain{padding:0 1rem!important}
    .main-content{flex-direction:column!important;padding:0 1rem!important}
    .upload-panel{border-right:1px solid var(--bdr)!important;border-bottom:none!important}
    .brutal-header{padding:2rem 1rem 1rem}
    .header-title{font-size:clamp(2rem,10vw,3.5rem)!important}
    .content-panel{min-height:auto!important}
    #output-terminal textarea{min-height:280px!important}
    .brutal-footer{padding:2rem 1rem}
}
</style>
""")
        
        # ═══════════════════════════════════════════
        # HERO HEADER
        # ═══════════════════════════════════════════
        gr.HTML("""
        <div class="brutal-header">
            <div class="header-rule"></div>
            <h1 class="header-title">MEETING<br>ASSISTANT</h1>
            <div class="header-subtitle">
                <span class="subtitle-dot"></span>
                AI-POWERED ANALYSIS ENGINE
                <span class="subtitle-dot"></span>
            </div>
            <div class="header-rule"></div>
            <p class="header-desc">
                UPLOAD MEETING AUDIO (UP TO 60 MINUTES) &mdash; RECEIVE <strong>TRANSCRIPTION</strong>,
                <strong>SUMMARY</strong>, <strong>TASK LIST</strong>
                &amp; <strong>SENTIMENT ANALYSIS</strong>
            </p>
        </div>
        """)
        
        # ═══════════════════════════════════════════
        # MAIN CONTENT
        # ═══════════════════════════════════════════
        with gr.Row(elem_classes=["main-content", "stagger-in"]):
            with gr.Column(scale=11, elem_classes=["content-panel", "upload-panel"]):
                gr.HTML("""<div class="panel-label"><span class="panel-label-arrow">&gt;</span> INPUT</div>""")
                gr.HTML("""<div id="upload-status" class="upload-status">
                    <span class="status-dot"></span>
                    <span class="status-text">AWAITING INPUT</span>
                </div>""")
                
                gr.HTML("""<div id="processing-indicator" class="processing-indicator">
                    <span class="status-dot"></span>
                    <span class="status-text">PROCESSING...</span>
                    <div class="progress-bar"></div>
                </div>""")
                
                audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    show_label=False,
                    elem_id="audio-upload"
                )
                
                gr.Examples(
                    examples=[[_demo_audio_path]],
                    inputs=audio_input,
                    label="SAMPLE AUDIO",
                    examples_per_page=1
                )
                
                with gr.Row(elem_classes=["button-row"]):
                    clear_btn = gr.Button("CLEAR", elem_id="btn-clear", variant="secondary", size="lg")
                    submit_btn = gr.Button("PROCESS", elem_id="btn-submit", variant="primary", size="lg")
            
            with gr.Column(scale=9, elem_classes=["content-panel", "output-panel"]):
                gr.HTML("""<div class="panel-label"><span class="panel-label-arrow">&gt;</span> OUTPUT</div>""")
                
                output_display = gr.Textbox(
                    label="",
                    lines=22,
                    max_lines=35,
                    show_label=False,
                    interactive=False,
                    placeholder="AWAITING INPUT...",
                    elem_id="output-terminal"
                )
                
                gr.HTML("""<div class="download-section">
                    <span class="panel-label-arrow">&gt;</span> EXPORT
                </div>""")
                
                download_file = gr.File(
                    label="",
                    elem_id="download-file",
                    visible=True,
                    interactive=False
                )
        
        # ═══════════════════════════════════════════
        # EVENT HANDLERS
        # ═══════════════════════════════════════════
        submit_btn.click(
            fn=process_meeting_audio,
            inputs=[audio_input],
            outputs=[output_display, download_file],
            js="""
            function(...args) {
                var audioEl = document.querySelector('#audio-upload audio');
                var hasAudio = audioEl && audioEl.src && audioEl.src !== window.location.href;
                if (hasAudio && window.__uploadStatus) {
                    window.__uploadStatus.set('processing', 'PROCESSING...');
                    var indicator = document.getElementById('processing-indicator');
                    if (indicator) {
                        indicator.classList.add('active');
                    }
                }
                return args;
            }
            """
        )
        
        clear_btn.click(
            fn=clear_interface,
            inputs=[],
            outputs=[audio_input, output_display, download_file],
            js="""
            function() {
                var indicator = document.getElementById('processing-indicator');
                if (indicator) {
                    indicator.classList.remove('active');
                }
                return [];
            }
            """
        )
        
        # ═══════════════════════════════════════════
        # FOOTER
        # ═══════════════════════════════════════════
        gr.HTML("""
        <footer class="brutal-footer">
            <div class="footer-rule"></div>
            <p class="footer-text">
                MODELS: <span>WHISPER</span> &middot; <span>BART</span> &middot; <span>ROBERTA</span> &middot; <span>LANGCHAIN</span>
                &nbsp;&mdash;&nbsp;
                CREATED BY <span>POUYADEVA1</span> &middot; OPEN SOURCE &middot; MIT LICENSE
            </p>
        </footer>
        """)
    
    return interface


def _generate_demo_audio():
    """Ensure the demo meeting WAV file exists — download from HF Hub or use embedded fallback."""
    demo_path = os.path.join(os.path.dirname(__file__), "sample_meeting.wav")
    if os.path.exists(demo_path) and os.path.getsize(demo_path) > 1000:
        return demo_path

    # Try downloading from HF Hub dataset
    try:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN")
        downloaded = hf_hub_download(
            repo_id="PouyaDevA1/ai-meeting-samples",
            filename="sample_meeting.wav",
            repo_type="dataset",
            token=hf_token,
            local_dir=os.path.dirname(demo_path),
        )
        # hf_hub_download may return a different path; rename to expected
        if downloaded != demo_path and os.path.exists(downloaded):
            import shutil
            shutil.move(downloaded, demo_path)
        logger.info(f"✅ Demo audio downloaded from HF Hub ({os.path.getsize(demo_path)} bytes)")
        return demo_path
    except Exception as e:
        logger.warning(f"⚠️ Could not download from HF Hub: {e}")

    # Fallback: try embedded audio data
    try:
        from sample_audio_data import get_demo_audio
        audio_bytes = get_demo_audio()
        with open(demo_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"✅ Demo audio extracted from embedded data ({len(audio_bytes)} bytes)")
        return demo_path
    except Exception as e:
        logger.warning(f"⚠️ Could not load embedded audio: {e} — generating tones")

    # Last resort: generate synthetic audio
    return _generate_synthetic_audio(demo_path)


def _generate_synthetic_audio(path):
    """Fallback: generate simple multi-speaker tone WAV file."""
    import math, random
    sample_rate = 16000
    duration = 12.0
    n_frames = int(sample_rate * duration)

    def _env(t, start, end, attack=0.05, release=0.05):
        if t < start or t > end:
            return 0.0
        rel = t - start
        dur = end - start
        if rel < attack:
            return rel / attack
        if rel > dur - release:
            return max(0.0, (dur - rel) / release)
        return 1.0

    speakers = [(440.0, 0.0, 2.5), (554.0, 3.0, 5.5), (659.0, 6.0, 8.5), (440.0, 9.0, 11.0)]
    audio_data = []
    for i in range(n_frames):
        t = i / sample_rate
        s = 0.0
        for freq, start, end in speakers:
            e = _env(t, start, end)
            if e > 0:
                s += e * 0.3 * math.sin(2 * math.pi * freq * t)
                s += e * 0.1 * math.sin(2 * math.pi * freq * 2 * t)
        s += (random.random() - 0.5) * 0.01
        audio_data.append(int(max(-1.0, min(1.0, s)) * 32767))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.setnframes(n_frames)
        wf.setcomptype("NONE", "not compressed")
        wf.writeframes(struct.pack("<" + "h" * len(audio_data), *audio_data))
    return path


# Ensure demo audio exists before initializing the assistant
_demo_audio_path = _generate_demo_audio()

# Initialize the meeting assistant
meeting_assistant = MeetingAssistant()

# Initialize and launch with speed optimizations
demo = create_interface()
# Set theme and js as attributes (Gradio 5+ compat — also set in Blocks() ctor)
demo.theme = gr.themes.Base()
demo.js = custom_js
demo.queue(default_concurrency_limit=3)
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
)
