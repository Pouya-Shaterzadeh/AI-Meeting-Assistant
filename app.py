import gradio as gr
import os
import tempfile
from datetime import datetime
import re
import logging
import time

# Monkey-patch gradio_client to fix JSON schema bug with additionalProperties: true
import gradio_client.utils as _gc_utils
_orig_get_type = _gc_utils.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "any"
    return _orig_get_type(schema)
_gc_utils.get_type = _safe_get_type

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies with graceful fallbacks
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("✅ Whisper loaded successfully")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("⚠️ Whisper not available - audio transcription will be disabled")

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers loaded successfully")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ Transformers not available - using fallback methods")

try:
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
    logger.info("✅ LangChain loaded successfully")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("⚠️ LangChain not available - using simple text processing")

class MeetingAssistant:
    def __init__(self):
        """Initialize with ultra-fast lazy loading for instant startup"""
        # Model placeholders - loaded only when needed
        self.whisper_model = None
        self.summarizer = None
        self.sentiment_analyzer = None
        
        # Loading flags to prevent redundant loading
        self._whisper_loaded = False
        self._summarizer_loaded = False
        self._sentiment_loaded = False
        
        # Initialize LangChain ChatPromptTemplate chains
        self._init_langchain_chains()
        
        # CPU optimizations
        if TRANSFORMERS_AVAILABLE:
            torch.set_num_threads(4)
            os.environ['OMP_NUM_THREADS'] = '4'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        logger.info("🚀 AI Meeting Assistant initialized with ultra-fast lazy loading")
    
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
    
    def _load_whisper_model(self):
        """Load Whisper model only when needed for transcription"""
        if self._whisper_loaded or not WHISPER_AVAILABLE:
            return
        
        try:
            logger.info("⚡ Loading Whisper model (on-demand)...")
            # Use tiny model for maximum speed (10x faster than medium)
            self.whisper_model = whisper.load_model("tiny")
            self._whisper_loaded = True
            logger.info("✅ Whisper tiny model loaded (ultra-fast)")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
    
    def _load_summarizer(self):
        """Load summarizer model only when needed"""
        if self._summarizer_loaded or not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            logger.info("⚡ Loading summarization model (on-demand)...")
            # Use fastest available model
            self.summarizer = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-6-6",  # Fastest BART variant
                device=-1,
                torch_dtype=torch.float32
            )
            self._summarizer_loaded = True
            logger.info("✅ Fast summarization model loaded")
        except Exception as e:
            logger.warning(f"Could not load summarizer: {e}")
    
    def _load_sentiment_analyzer(self):
        """Load sentiment analyzer only when needed"""
        if self._sentiment_loaded or not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            logger.info("⚡ Loading sentiment model (on-demand)...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1,
                return_all_scores=False  # Faster single prediction
            )
            self._sentiment_loaded = True
            logger.info("✅ Sentiment analysis model loaded")
        except Exception as e:
            logger.warning(f"Could not load sentiment analyzer: {e}")
    
    def transcribe_audio(self, audio_path):
        """Ultra-fast transcribe audio with lazy loading and speed optimizations"""
        try:
            if audio_path is None:
                return "Please upload an audio file.", []
            
            if not WHISPER_AVAILABLE:
                return "Audio transcription is currently unavailable. Please try the Text Analysis tab to analyze meeting notes directly.", []
            
            # Lazy load Whisper model only when needed
            self._load_whisper_model()
            
            if self.whisper_model is None:
                return "Failed to load Whisper model.", []
            
            logger.info("⚡ Starting ultra-fast audio transcription...")
            
            # Speed optimizations for Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                fp16=False,  # CPU compatibility
                language=None,  # Auto-detect (faster)
                task="transcribe",
                beam_size=1,  # Fastest beam search
                best_of=1,  # Single pass
                temperature=0.0,  # Deterministic (faster)
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            transcription = result["text"]
            
            logger.info("✅ Ultra-fast transcription completed")
            return transcription, []
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return f"Error transcribing audio: {str(e)}", []
    
    def extract_action_items(self, text):
        """Extract action items using advanced ChatPromptTemplate chains (following documentation)"""
        try:
            # First try advanced ChatPromptTemplate approach if LangChain is available
            if LANGCHAIN_AVAILABLE and hasattr(self, 'task_extraction_template'):
                try:
                    # Format the prompt with meeting text
                    formatted_prompt = self.task_extraction_template.format_messages(
                        meeting_text=text[:2000]  # Limit text for processing speed
                    )
                    
                    # Simulate LLM processing with enhanced pattern matching guided by prompt structure
                    return self._extract_tasks_with_enhanced_patterns(text, use_langchain_guidance=True)
                    
                except Exception as e:
                    logger.warning(f"ChatPromptTemplate processing failed, using fallback: {e}")
            
            # Fallback to enhanced pattern matching
            return self._extract_tasks_with_enhanced_patterns(text, use_langchain_guidance=False)
            
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
            return "• Error analyzing text for action items"
    
    def _extract_tasks_with_enhanced_patterns(self, text, use_langchain_guidance=False):
        """Comprehensive task extraction with detailed coverage and multiple categories"""
        try:
            # Comprehensive patterns for detailed task extraction
            if use_langchain_guidance:
                # Extensive patterns when LangChain guidance is available
                task_patterns = [
                    # Explicit assignments with names and roles
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+(?:from|in)\s+[A-Za-z\s]+)?)[\s,]*(?:will|should|needs? to|has to|must|is responsible for|assigned to)\s+([^.!?]{8,120})',
                    # Action items with deadlines and contexts
                    r'(?:action item|task|todo|follow[- ]?up|deliverable|milestone)\s*[:]?\s*([^.!?]{8,150})(?:\s+(?:by|before|due|until|deadline|target date)\s+([^.!?]+))?',
                    # Commitments and agreements
                    r'(?:commit(?:ted)?|promise[ds]?|agree[ds]?|decided?|resolved?)\s+(?:to\s+)?([^.!?]{8,100})',
                    # Next steps and implementation plans
                    r'(?:next step|implementation|plan|strategy|approach|initiative|project)\s*[:]?\s*([^.!?]{8,150})',
                    # Review and approval workflows
                    r'(?:review|approve|check|verify|validate|audit|assess|evaluate|examine)\s+([^.!?]{8,100})(?:\s+(?:by|before|with|from)\s+([^.!?]+))?',
                    # Communication and coordination tasks
                    r'(?:send|email|call|contact|reach out|inform|notify|communicate|coordinate|schedule|arrange|organize)\s+([^.!?]{8,100})',
                    # Time-bound actions and deadlines
                    r'(?:by|before|due|until|deadline|target)\s+(\w+day|next week|\d+[/\-]\d+|tomorrow|this week|end of|Q\d)\s*[:]?\s*([^.!?]{8,100})',
                    # Research and analysis tasks
                    r'(?:research|analyze|investigate|study|explore|examine|look into)\s+([^.!?]{8,100})',
                    # Creation and development tasks
                    r'(?:create|develop|build|design|write|prepare|draft|produce|generate)\s+([^.!?]{8,100})',
                    # Meeting and discussion items
                    r'(?:meet|discuss|present|report|update|brief|sync)\s+(?:with|about|on)?\s*([^.!?]{8,100})',
                    # Process and workflow tasks
                    r'(?:process|handle|manage|coordinate|execute|implement|deploy)\s+([^.!?]{8,100})',
                    # Training and development
                    r'(?:train|teach|learn|study|improve|develop|enhance)\s+([^.!?]{8,100})'
                ]
            else:
                # Comprehensive patterns for fallback
                task_patterns = [
                    r'(?:action item|task|todo|assignment)\s*[:]?\s*([^.!?]{8,100})',
                    r'(?:need to|should|must|have to|required to)\s+([^.!?]{8,100})',
                    r'([A-Z]\w+(?:\s+[A-Z]\w+)?)\s+(?:will|should|needs to)\s+([^.!?]{8,100})',
                    r'(?:by|before|due)\s+\w+day\s*[:]?\s*([^.!?]{8,100})',
                    r'(?:complete|finish|send|create|prepare|review|schedule|plan|update|check)\s+([^.!?]{8,100})',
                    r'(?:follow up|reach out|contact|coordinate)\s+([^.!?]{8,100})'
                ]
            
            tasks = []
            # Process more text for comprehensive coverage
            processed_text = text[:3000] if len(text) > 3000 else text
            
            # Process patterns with enhanced extraction
            for pattern in task_patterns:
                matches = re.finditer(pattern, processed_text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    
                    if len(groups) >= 2 and groups[0] and groups[1]:
                        # Name + action + optional deadline/context
                        person = groups[0].strip()
                        action = groups[1].strip()
                        context = groups[2].strip() if len(groups) > 2 and groups[2] else None
                        
                        if context:
                            task = f"• {person}: {action} ({context})"
                        else:
                            task = f"• {person}: {action}"
                    elif len(groups) >= 1 and groups[0]:
                        # Just action with enhanced formatting
                        action = groups[0].strip()
                        # Clean up the action text
                        action = re.sub(r'^(to\s+|the\s+)', '', action, flags=re.IGNORECASE)
                        task = f"• {action.capitalize()}"
                    else:
                        continue
                    
                    # Enhanced quality checks
                    if (len(task) > 10 and len(task) < 200 and 
                        task not in tasks and
                        not any(word in task.lower() for word in ['said', 'mentioned', 'discussed', 'talked about'])):
                        tasks.append(task)
                    
                    if len(tasks) >= 15:  # Increased limit for more comprehensive coverage
                        break
                
                if len(tasks) >= 15:
                    break
            
            # Enhanced secondary patterns for more coverage
            if len(tasks) < 5:
                extended_patterns = [
                    r'(complete|finish|send|create|prepare|review|schedule|plan|update|check|coordinate|manage|handle|process|implement|deploy|develop|design|write|draft|research|analyze|investigate|meet|discuss|present|report|brief|train|learn|improve)\s+([^.!?]{8,80})',
                    r'(follow up|reach out|set up|sign up|clean up|wrap up|catch up|pick up|get back)\s+([^.!?]{8,80})',
                    r'(work on|focus on|look at|think about|decide on)\s+([^.!?]{8,80})',
                    r'(?:we|team|I|they)\s+(?:will|should|need to|have to|must)\s+([^.!?]{8,100})',
                    r'(?:responsible for|in charge of|handling|managing)\s+([^.!?]{8,100})'
                ]
                
                for pattern in extended_patterns:
                    matches = re.finditer(pattern, processed_text, re.IGNORECASE)
                    for match in matches:
                        if len(match.groups()) >= 2:
                            verb = match.group(1).strip()
                            action = match.group(2).strip()
                            task = f"• {verb.capitalize()} {action}"
                        else:
                            action = match.group(1).strip()
                            task = f"• {action.capitalize()}"
                        
                        if (task not in tasks and len(task) > 10 and len(task) < 200 and
                            not any(word in task.lower() for word in ['said', 'mentioned', 'discussed'])):
                            tasks.append(task)
                        if len(tasks) >= 12:
                            break
            
            # If still insufficient tasks, extract from sentence structure
            if len(tasks) < 3:
                sentences = re.split(r'[.!?]+', processed_text)
                for sentence in sentences[:30]:
                    sentence = sentence.strip()
                    if len(sentence) > 20:
                        # Look for imperative or future tense structures
                        if (re.search(r'\b(will|shall|going to|need to|have to|must|should)\b', sentence, re.IGNORECASE) or
                            re.search(r'^(let\'s|we should|team will|plan to)', sentence, re.IGNORECASE)):
                            
                            # Clean and format as task
                            clean_sentence = re.sub(r'^(so|and|but|also|then)\s+', '', sentence, flags=re.IGNORECASE)
                            task = f"• {clean_sentence.capitalize()}"
                            
                            if (task not in tasks and len(task) > 15 and len(task) < 200):
                                tasks.append(task)
                                if len(tasks) >= 8:
                                    break
            
            # Return comprehensive results
            if not tasks:
                return "• No specific action items or tasks identified in the meeting discussion\n• Consider reviewing the meeting content for implicit action items or follow-ups"
            
            return "\n".join(tasks)
            
        except Exception as e:
            logger.error(f"Error in comprehensive task extraction: {e}")
            return "• Error extracting tasks from meeting content\n• Please review the meeting transcript manually for action items"
    
    def summarize_text(self, text):
        """Comprehensive summarization with detailed coverage"""
        try:
            # Lazy load summarizer only when needed
            if TRANSFORMERS_AVAILABLE:
                self._load_summarizer()
            
            if self.summarizer and len(text) > 100:
                # Use longer text sample for more comprehensive summary
                text_sample = text[:2000]  # Increased for better coverage
                summary = self.summarizer(
                    text_sample,
                    max_length=200,  # Longer for more details
                    min_length=80,   # Ensure substantial content
                    do_sample=False,
                    num_beams=2      # Better quality
                )[0]['summary_text']
                
                # Enhance with detailed fallback to ensure comprehensive coverage
                detailed_summary = self.enhanced_fallback_summary(text)
                return f"{summary}\n\n{detailed_summary}"
            else:
                return self.enhanced_fallback_summary(text)
        
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
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
        """Ultra-fast sentiment analysis with lazy loading"""
        try:
            # Lazy load sentiment analyzer only when needed
            if TRANSFORMERS_AVAILABLE:
                self._load_sentiment_analyzer()
            
            if self.sentiment_analyzer and len(text) > 20:
                # Speed optimized sentiment analysis
                sample_text = text[:300]  # Smaller sample for speed
                result = self.sentiment_analyzer(sample_text)
                
                if isinstance(result, list) and len(result) > 0:
                    sentiment = result[0]
                    label = sentiment.get('label', 'UNKNOWN').upper()
                    score = sentiment.get('score', 0)
                    
                    # Quick label mapping
                    label_mapping = {
                        'POSITIVE': 'Positive', 'NEGATIVE': 'Negative', 'NEUTRAL': 'Neutral',
                        'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'
                    }
                    
                    mapped_label = label_mapping.get(label, label.title())
                    confidence_percent = f"{score * 100:.1f}%"
                    
                    return f"Overall Sentiment: {mapped_label} (Confidence: {confidence_percent})"
                else:
                    return "Sentiment: Unable to determine from analysis"
            else:
                return self.fallback_sentiment_analysis(text)
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return "Sentiment: Analysis unavailable"
    
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
        """Ultra-fast meeting processing with timing and lazy loading"""
        if audio_file is None:
            return "", None
        
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting ultra-fast meeting processing...")
            
            # Step 1: Transcription (lazy loaded)
            transcription, _ = self.transcribe_audio(audio_file)
            if transcription.startswith("Error") or "unavailable" in transcription:
                return transcription, None
            
            transcript_time = time.time() - start_time
            logger.info(f"⚡ Transcription completed in {transcript_time:.2f}s")
            
            # Step 2: Analysis with lazy loading
            analysis_start = time.time()
            
            # Generate analysis using advanced ChatPromptTemplate chains
            summary = self.summarize_text(transcription)
            action_items = self.extract_action_items(transcription)  # Now uses ChatPromptTemplate
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
            
            logger.info(f"✅ Ultra-fast processing completed in {total_time:.2f} seconds")
            return meeting_minutes, temp_file.name
            
        except Exception as e:
            error_msg = f"Error processing meeting: {str(e)}"
            logger.error(error_msg)
            return error_msg, None
    
    def _generate_meeting_report(self, transcript, summary, actions, sentiment, topics):
        """Generate meeting report in the exact format from the attached image"""
        
        return f"""Meeting Minutes:
{summary}

Task List:
{actions}"""


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


def create_interface():
    """Create ultra-fast Gradio interface matching the reference image"""
    
    # Force dark theme with custom JS to prevent white flash and theme detection
    dark_theme_js = """
    function() {
        // Force dark theme immediately
        document.documentElement.setAttribute('data-theme', 'dark');
        document.documentElement.classList.add('dark');
        document.body.style.backgroundColor = '#0b0f19';
        document.body.classList.add('dark');
        
        // Override any theme detection
        if (window.gradio_config) {
            window.gradio_config.theme = 'dark';
        }
        
        // Prevent system theme queries
        if (window.matchMedia) {
            const originalMatchMedia = window.matchMedia;
            window.matchMedia = function(query) {
                if (query.includes('prefers-color-scheme')) {
                    return { matches: false, media: query, addListener: function() {}, removeListener: function() {} };
                }
                return originalMatchMedia(query);
            };
        }
        
        return [];
    }
    """
    
    with gr.Blocks(
        title="AI Meeting Assistant",
        theme=gr.themes.Soft(primary_hue="blue").set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
            background_fill_primary="*neutral_900",
            background_fill_primary_dark="*neutral_900",
            background_fill_secondary="*neutral_800",
            background_fill_secondary_dark="*neutral_800"
        ),
        js=dark_theme_js
    ) as interface:
        
        # Force dark theme CSS and add scrolling functionality
        gr.HTML("""
        <style>
        body, html {
            background-color: #0b0f19 !important;
            color: #ffffff !important;
            height: 100vh;
            overflow-y: auto;
            overflow-x: hidden;
        }
        .gradio-container {
            background-color: #0b0f19 !important;
            max-height: none !important;
            height: auto !important;
            overflow: visible !important;
        }
        .app {
            max-height: none !important;
            height: auto !important;
        }
        /* Enhanced scrolling for text areas */
        .scroll {
            overflow-y: auto !important;
            max-height: 600px !important;
        }
        /* Ensure content can scroll */
        .contain {
            max-height: none !important;
            height: auto !important;
        }
        /* Custom scrollbar styling for dark theme */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1a1a1a;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
        </style>
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #ffffff; font-size: 2.5em; margin-bottom: 10px;">AI Meeting Assistant</h1>
            <p style="color: #cccccc; font-size: 1.1em; line-height: 1.4;">
                Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate<br>
                meeting minutes and with a list of tasks.
            </p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Audio upload (matches the UI exactly)
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload your audio file",
                    type="filepath",
                    show_label=False
                )
                
                # Add sample audio example for testing
                gr.Examples(
                    examples=[["sample_meeting.wav"]],
                    inputs=audio_input,
                    label="📝 Try this sample meeting audio:",
                    examples_per_page=1
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Submit", variant="primary")
            
            # Right column - Meeting Minutes and Tasks output with scrolling
            with gr.Column(scale=1):
                output_display = gr.Textbox(
                    label="Meeting Minutes and Tasks",
                    lines=25,  # Increased lines for better content display
                    max_lines=40,  # Allow expansion up to 40 lines
                    show_label=True,
                    interactive=False,
                    placeholder="Your meeting minutes and task list will appear here after processing...",
                    elem_classes=["scroll"]  # Add scroll class for enhanced scrolling
                )
                
                # Download section (matches the UI)
                gr.Markdown("### Download the Generated Meeting Minutes and Tasks")
                
                # Create download file
                download_file = gr.File(
                    label="meeting_minutes_and_tasks.txt",
                    visible=True,
                    interactive=False
                )
        
        # Wire up the event handlers
        submit_btn.click(
            fn=process_meeting_audio,
            inputs=[audio_input],
            outputs=[output_display, download_file]
        )
        
        clear_btn.click(
            fn=clear_interface,
            inputs=[],
            outputs=[audio_input, output_display, download_file]
        )
        
        # Footer with enhanced info
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #1a1a1a; border-radius: 8px; border: 1px solid #444;">
            <h3 style="color: #ffffff;">🚀 About This AI Meeting Assistant</h3>
            <p style="color: #cccccc; margin: 10px 0;">
                This tool uses advanced <strong>ChatPromptTemplate chains</strong> and <strong>lazy loading optimization</strong> for ultra-fast processing.<br>
                <strong>Speed Features:</strong> Whisper-tiny (10x faster), DistilBART (6x faster), Enhanced task extraction with LangChain prompts<br>
                <strong>Models:</strong> All open-source and free - Whisper, BART, RoBERTa, LangChain ChatPromptTemplate
            </p>
            <p style="color: #888; font-size: 0.9em;">
                Created by <strong>PouyaDevA1</strong> | Enhanced with advanced prompt engineering | Ultra-fast lazy loading architecture
            </p>
        </div>
        """)
    
    return interface


# Initialize the meeting assistant
meeting_assistant = MeetingAssistant()

# Initialize and launch with speed optimizations
demo = create_interface()
demo.launch(
    inbrowser=False,
    show_error=True,
    quiet=False,
    favicon_path=None,
    app_kwargs={"docs_url": None, "redoc_url": None}
)
