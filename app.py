import gradio as gr
import os
import tempfile
from datetime import datetime
import re
import logging
import time

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

    # Model IDs — best-in-class serverless models
    ASR_MODEL = "openai/whisper-large-v3-turbo"
    SUM_MODEL = "facebook/bart-large-cnn"
    SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

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

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            result = self.client.automatic_speech_recognition(
                audio_bytes,
                model=self.ASR_MODEL
            )
            transcription = result.get("text", "")

            if not transcription:
                return "No speech detected in the audio file.", []

            logger.info("✅ Transcription completed via HF Inference API")
            return transcription, []

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return f"Error transcribing audio: {str(e)}", []
    
    def extract_action_items(self, text):
        """Extract action items via LLM (Mistral-7B) or fall back to regex"""
        try:
            if self.client:
                return self._extract_tasks_with_llm(text)
            else:
                return self._extract_tasks_fallback(text)
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
            return self._extract_tasks_fallback(text)

    def _extract_tasks_with_llm(self, text):
        """Use Mistral-7B to extract structured action items from transcript"""
        try:
            if LANGCHAIN_AVAILABLE and hasattr(self, 'task_extraction_template'):
                messages = self.task_extraction_template.format_messages(
                    meeting_text=text[:4000]
                )
                # Convert LangChain messages to dicts for InferenceClient
                api_messages = [
                    {"role": "system" if m.type == "system" else "user", "content": m.content}
                    for m in messages
                ]
            else:
                api_messages = [
                    {"role": "system", "content": "You are an expert meeting analyst. Extract actionable tasks with clear ownership, deadlines, and specifics. Format each task as a bullet point starting with '•'. Include WHO, WHAT, and WHEN for each task."},
                    {"role": "user", "content": f"Extract ALL actionable tasks from this meeting transcript:\n\n{text[:4000]}\n\nEXTRACTED TASKS:"}
                ]

            logger.info("⚡ Extracting action items via Mistral-7B...")
            response = self.client.chat_completion(
                messages=api_messages,
                model=self.LLM_MODEL,
                max_tokens=self._llm_max_tokens,
                temperature=0.2
            )
            content = response["choices"][0]["message"]["content"]
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
        """Summarize via HF Inference API (bart-large-cnn) or fallback"""
        try:
            if not self.client:
                return self.enhanced_fallback_summary(text)

            if len(text) < 100:
                return self.enhanced_fallback_summary(text)

            logger.info("⚡ Summarizing via HF Inference API (bart-large-cnn)...")
            result = self.client.summarization(
                text,
                model=self.SUM_MODEL,
                parameters={"max_length": 200, "min_length": 80, "do_sample": False}
            )
            summary = result.get("summary_text", "")
            logger.info("✅ Summarization completed via API")
            return summary

        except Exception as e:
            logger.warning(f"API summarization failed, falling back: {e}")
            return self.enhanced_fallback_summary(text)
    
    def enhanced_fallback_summary(self, text):
        """Comprehensive fallback summarization with detailed bullet points"""
        try:
            sentences = re.split(r'[.!?]+', text)
            
            # Expanded key phrases for comprehensive coverage
            key_categories = {
                'financial': ['revenue', 'profit', 'budget', 'cost', 'investment', 'financial', 'money', 'funding', 'quarter', 'forecast'],
                'business': ['strategy', 'business', 'market', 'customer', 'product', 'service', 'growth', 'expansion', 'competition'],
                'operational': ['process', 'operation', 'workflow', 'efficiency', 'performance', 'quality', 'timeline', 'deadline'],
                'decisions': ['decision', 'approve', 'reject', 'choose', 'select', 'agree', 'disagree', 'vote', 'consensus'],
                'updates': ['update', 'progress', 'status', 'report', 'achievement', 'milestone', 'completion', 'result'],
                'planning': ['plan', 'strategy', 'roadmap', 'goal', 'objective', 'target', 'initiative', 'project'],
                'issues': ['problem', 'issue', 'concern', 'risk', 'challenge', 'obstacle', 'difficulty', 'blocker'],
                'people': ['team', 'staff', 'employee', 'hire', 'promotion', 'training', 'resource', 'department']
            }
            
            categorized_points = {category: [] for category in key_categories}
            general_points = []
            
            # Process all sentences for comprehensive coverage
            for sentence in sentences[:50]:  # Analyze more sentences
                sentence = sentence.strip()
                if len(sentence) > 25:  # Slightly lower threshold
                    sentence_lower = sentence.lower()
                    
                    # Categorize sentences
                    categorized = False
                    for category, keywords in key_categories.items():
                        if any(keyword in sentence_lower for keyword in keywords):
                            if len(categorized_points[category]) < 3:  # Limit per category
                                categorized_points[category].append(sentence)
                                categorized = True
                                break
                    
                    # Add to general if not categorized and we need more content
                    if not categorized and len(general_points) < 5:
                        general_points.append(sentence)
            
            # Build comprehensive summary
            summary_parts = []
            
            for category, points in categorized_points.items():
                if points:
                    category_title = category.replace('_', ' ').title()
                    for point in points:
                        summary_parts.append(f"• {point}")
            
            # Add general points if we have space
            for point in general_points[:3]:
                summary_parts.append(f"• {point}")
            
            if summary_parts:
                return "\n".join(summary_parts)
            else:
                # Fallback: extract first substantial sentences
                substantial_sentences = [s.strip() for s in sentences[:8] if len(s.strip()) > 30]
                if substantial_sentences:
                    return "\n".join([f"• {s}" for s in substantial_sentences[:6]])
                else:
                    return f"• {text[:300]}..." if len(text) > 300 else f"• {text}"
        
        except Exception as e:
            logger.error(f"Error in enhanced summary: {str(e)}")
            return "• Comprehensive summary not available"
    
    def fallback_summary(self, text):
        """Maintain original method for compatibility"""
        return self.enhanced_fallback_summary(text)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment via HF Inference API (roberta-base-sentiment) on full transcript"""
        try:
            if not self.client:
                return self.fallback_sentiment_analysis(text)

            if len(text) < 20:
                return self.fallback_sentiment_analysis(text)

            logger.info("⚡ Analyzing sentiment via HF Inference API (full transcript)...")
            result = self.client.text_classification(
                text,
                model=self.SENT_MODEL
            )
            if isinstance(result, list) and len(result) > 0:
                sentiment = result[0]
                label = sentiment.get("label", "UNKNOWN").upper()
                score = sentiment.get("score", 0)
                label_map = {
                    "POSITIVE": "Positive", "NEGATIVE": "Negative", "NEUTRAL": "Neutral",
                    "LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive",
                }
                mapped = label_map.get(label, label.title())
                return f"Overall Sentiment: {mapped} (Confidence: {score * 100:.1f}%)"
            else:
                return self.fallback_sentiment_analysis(text)

        except Exception as e:
            logger.warning(f"API sentiment failed, falling back: {e}")
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
        """Fast key topics identification"""
        try:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 
                'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 
                'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with', 'have', 'this', 'will', 'your', 
                'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 
                'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 
                'would', 'there', 'could', 'other', 'after', 'first', 'never', 'these', 'think', 'where', 'being', 
                'every', 'great', 'might', 'shall', 'still', 'those', 'under', 'while', 'again', 'before', 'right', 
                'about', 'also', 'back', 'call', 'came', 'each', 'even', 'going', 'look', 'most', 'move', 'need', 
                'only', 'said', 'same', 'show', 'tell', 'turn', 'ways', 'went', 'work', 'year', 'meeting'
            }
            
            word_count = {}
            for word in words:
                if word not in stop_words and len(word) > 3:
                    word_count[word] = word_count.get(word, 0) + 1
            
            if not word_count:
                return "• No significant topics identified"
            
            top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]
            topics = [f"• {word.capitalize()} (mentioned {count} times)" for word, count in top_words if count > 1]
            
            return "\n".join(topics) if topics else "• No recurring topics identified"
        
        except Exception as e:
            logger.error(f"Error identifying topics: {str(e)}")
            return "• Error analyzing topics"
    
    def process_meeting_simple(self, audio_file):
        """Process meeting audio via HF Inference API (serverless pipeline)"""
        if audio_file is None:
            return "", None
        
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting meeting processing via HF Inference API...")
            
            # Step 1: Transcription (whisper-large-v3-turbo via API)
            transcription, _ = self.transcribe_audio(audio_file)
            if transcription.startswith("Error") or transcription.startswith("HF_TOKEN") or "unavailable" in transcription:
                return transcription, None
            
            transcript_time = time.time() - start_time
            logger.info(f"⚡ Transcription completed in {transcript_time:.2f}s")
            
            # Step 2: Analysis (all via API, fallback to local methods on failure)
            analysis_start = time.time()
            
            summary = self.summarize_text(transcription)
            action_items = self.extract_action_items(transcription)
            sentiment = self.analyze_sentiment(transcription)
            key_topics = self.identify_key_topics(transcription)
            
            analysis_time = time.time() - analysis_start
            logger.info(f"⚡ Analysis completed in {analysis_time:.2f}s")
            
            # Step 3: Generate comprehensive report
            meeting_minutes = self._generate_meeting_report(
                transcription, summary, action_items, sentiment, key_topics
            )
            
            total_time = time.time() - start_time
            
            # Create download file
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', prefix='meeting_minutes_')
            temp_file.write(meeting_minutes)
            temp_file.close()
            
            logger.info(f"✅ Processing completed in {total_time:.2f}s (serverless)")
            return meeting_minutes, temp_file.name
            
        except Exception as e:
            error_msg = f"Error processing meeting: {str(e)}"
            logger.error(error_msg)
            return error_msg, None
    
    def _generate_meeting_report(self, transcript, summary, actions, sentiment, topics):
        """Generate meeting report with brutalist terminal formatting"""
        sep = "=" * 52
        return f"""{sep}
                  MEETING ANALYSIS REPORT
{sep}

>> SUMMARY
{summary}

>> TASK LIST
{actions}

>> SENTIMENT
{sentiment}

>> KEY TOPICS
{topics}

{sep}
          PROCESSED BY AI MEETING ASSISTANT
{sep}"""


def process_meeting_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file to analyze.", None

    meeting_report, temp_file = meeting_assistant.process_meeting_simple(audio_file)

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
       UPLOAD STATUS — global helpers + document-level listeners
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

    /* Detect file selection in #audio-upload (capture phase — fires before Gradio) */
    document.addEventListener('change', function(e) {
        var t = e.target;
        if (t && t.tagName === 'INPUT' && t.type === 'file' && t.closest && t.closest('#audio-upload') && t.files && t.files.length > 0) {
            window.__uploadStatus.set('uploading', 'UPLOADING...');
        }
    }, true);

    /* Poll for <audio> element inside #audio-upload — upload/recording complete */
    var __audioCheck = setInterval(function() {
        var audio = document.querySelector('#audio-upload audio');
        if (audio && audio.src && audio.src !== window.location.href && audio.readyState >= 2) {
            window.__uploadStatus.set('complete', 'UPLOAD COMPLETE');
        }
    }, 600);

    /* CLEAR button — reset status */
    document.addEventListener('click', function(e) {
        if (e.target && e.target.closest && e.target.closest('#btn-clear')) {
            setTimeout(function() { window.__uploadStatus.set('', 'AWAITING INPUT'); }, 150);
        }
    });

    /* Watch output terminal for processing completion */
    var __termReady = setInterval(function() {
        var term = document.getElementById('output-terminal');
        if (!term) return;
        clearInterval(__termReady);
        var obs = new MutationObserver(function() {
            var ta = term.querySelector('textarea');
            if (ta && ta.value && ta.value.trim().length > 0
                && ta.value.indexOf('Error') !== 0
                && ta.value.indexOf('Please upload') !== 0) {
                window.__uploadStatus.set('complete', 'PROCESSING COMPLETE');
                setTimeout(function() { window.__uploadStatus.set('', 'AWAITING INPUT'); }, 3500);
            }
        });
        obs.observe(term, { childList: true, subtree: true, characterData: true });
    }, 300);
    
    return [];
}
"""


def create_interface():
    """Create brutalist-industrial Gradio interface"""
    
    with gr.Blocks(
        title="AI Meeting Assistant",
        theme=gr.themes.Base(),
        js=custom_js
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
.gradio-container{max-width:100%!important;padding:0!important;background:transparent!important;margin:0!important}
.gradio-container .contain{max-width:1100px!important;margin:0 auto!important;padding:0 2.5rem!important}
.gradio-container .app{background:transparent!important}
.gradio-container .main{background:transparent!important}
.gr-box,.gr-panel,.gr-form{background:transparent!important;border:none!important;box-shadow:none!important;border-radius:0!important}
.prose{color:var(--txt)!important}
.prose *{color:var(--txt)!important}

/* Hide Gradio branding only (not our custom footer) */
gradio-app footer:not(.brutal-footer),#footer,.built-with-gradio{display:none!important;visibility:hidden!important;height:0!important;overflow:hidden!important;opacity:0!important}

/* ═══ GRADIO LABEL OVERRIDES ═══ */
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
    text-align:center;
    padding:3.5rem 2.5rem 1.5rem;
    max-width:1100px;
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
    margin:0 auto;
}
.header-desc strong{color:var(--txt2);font-weight:500}

/* ═══ MAIN CONTENT GRID ═══ */
.main-content{
    display:flex!important;
    gap:0!important;
    max-width:1100px!important;
    margin:0 auto!important;
    padding:0 2.5rem!important;
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
    padding:0.5rem 0.75rem;
    margin-bottom:0.75rem;
    background:var(--bg3);
    border:1px solid var(--bdr);
    font-family:var(--fm);
    font-size:0.65rem;
    letter-spacing:0.1em;
    text-transform:uppercase;
    color:var(--txt2);
    transition:all var(--t);
}
.upload-status.uploading{
    border-color:var(--acc);
    color:var(--acc);
    background:rgba(255,107,53,0.06);
}
.upload-status.processing{
    border-color:var(--acc);
    color:var(--acc);
    background:rgba(255,107,53,0.08);
}
.upload-status.complete{
    border-color:#22c55e;
    color:#22c55e;
    background:rgba(34,197,94,0.06);
}
.status-dot{
    width:8px;
    height:8px;
    border-radius:50%;
    background:currentColor;
    flex-shrink:0;
    opacity:0.6;
    transition:all var(--t);
}
.upload-status.uploading .status-dot,
.upload-status.processing .status-dot{
    opacity:1;
    animation:statusPulse 0.8s ease-in-out infinite;
}
.upload-status.complete .status-dot{
    opacity:1;
    background:#22c55e;
}
@keyframes statusPulse{
    0%,100%{transform:scale(1);opacity:1}
    50%{transform:scale(1.6);opacity:0.4}
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
.brutal-footer{text-align:center;padding:2.5rem 2.5rem 3rem;max-width:1100px;margin:0 auto}
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
                UPLOAD MEETING AUDIO &mdash; RECEIVE <strong>TRANSCRIPTION</strong>,
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
                
                audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    show_label=False,
                    elem_id="audio-upload"
                )
                
                gr.Examples(
                    examples=[["sample_meeting.wav"]],
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
            function() {
                var audioEl = document.querySelector('#audio-upload audio');
                var hasAudio = audioEl && audioEl.src && audioEl.src !== window.location.href;
                if (hasAudio && window.__uploadStatus) {
                    window.__uploadStatus.set('processing', 'PROCESSING...');
                }
                return [];
            }
            """
        )
        
        clear_btn.click(
            fn=clear_interface,
            inputs=[],
            outputs=[audio_input, output_display, download_file]
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


# Initialize the meeting assistant
meeting_assistant = MeetingAssistant()

# Initialize and launch with speed optimizations
demo = create_interface()
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    inbrowser=False,
    show_error=True,
    quiet=False,
    favicon_path=None,
    footer_links=[],
    app_kwargs={"docs_url": None, "redoc_url": None}
)
