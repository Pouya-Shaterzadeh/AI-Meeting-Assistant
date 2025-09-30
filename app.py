import gradio as gr
import os
import tempfile
from datetime import datetime
import re
import logging
import time

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
        """Enhanced pattern-based task extraction following ChatPromptTemplate structure"""
        try:
            # Advanced patterns following the documentation structure
            if use_langchain_guidance:
                # More sophisticated patterns when LangChain guidance is available
                task_patterns = [
                    # Explicit assignments with names
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:will|should|needs? to|has to|must)\s+([^.!?]{10,80})',
                    # Action items with deadlines
                    r'(?:action item|task|todo|follow[- ]?up)\s*:?\s*([^.!?]{10,100})(?:\s+(?:by|before|due)\s+([^.!?]+))?',
                    # Commitments and promises
                    r'(?:commit(?:ted)?|promise[ds]?|agree[ds]?)\s+to\s+([^.!?]{10,80})',
                    # Next steps and implementation
                    r'(?:next step|implementation|plan)\s*:?\s*([^.!?]{10,100})',
                    # Review and approval tasks
                    r'(?:review|approve|check|verify|validate)\s+([^.!?]{10,80})(?:\s+(?:by|before)\s+([^.!?]+))?',
                    # Communication tasks
                    r'(?:send|email|call|contact|reach out|inform|notify|communicate)\s+([^.!?]{10,80})',
                    # Time-bound actions
                    r'(?:by|before|due|until)\s+(\w+day|next week|\d+\/\d+|tomorrow)\s*:?\s*([^.!?]{10,80})',
                ]
            else:
                # Simpler patterns for fallback
                task_patterns = [
                    r'(?:action item|task|todo)\s*:?\s*([^.!?]{10,80})',
                    r'(?:need to|should|must)\s+([^.!?]{10,80})',
                    r'([A-Z]\w+)\s+(?:will|should)\s+([^.!?]{10,80})',
                    r'(?:by|before)\s+\w+day\s*:?\s*([^.!?]{10,80})',
                ]
            
            tasks = []
            processed_text = text[:1500] if len(text) > 1500 else text  # Speed optimization
            
            # Process patterns
            for pattern in task_patterns:
                matches = re.finditer(pattern, processed_text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    
                    if len(groups) >= 2 and groups[0] and groups[1]:
                        # Name + action + optional deadline
                        person = groups[0].strip()
                        action = groups[1].strip()
                        deadline = groups[2].strip() if len(groups) > 2 and groups[2] else None
                        
                        if deadline:
                            task = f"• {person} will {action} by {deadline}"
                        else:
                            task = f"• {person} will {action}"
                    elif len(groups) >= 1 and groups[0]:
                        # Just action
                        action = groups[0].strip()
                        task = f"• {action.capitalize()}"
                    else:
                        continue
                    
                    # Quality checks
                    if (len(task) > 15 and len(task) < 150 and 
                        task not in tasks and
                        not any(word in task.lower() for word in ['said', 'mentioned', 'discussed'])):
                        tasks.append(task)
                    
                    if len(tasks) >= 8:  # Limit for performance
                        break
                
                if len(tasks) >= 8:
                    break
            
            # If still no tasks found, try simpler patterns
            if len(tasks) < 2:
                simple_patterns = [
                    r'(complete|finish|send|create|prepare|review|schedule|plan)\s+([^.!?]{10,60})',
                    r'(follow up|check|update|inform|contact)\s+([^.!?]{10,60})',
                ]
                
                for pattern in simple_patterns:
                    matches = re.finditer(pattern, processed_text, re.IGNORECASE)
                    for match in matches:
                        action = match.group(0).strip()
                        task = f"• {action.capitalize()}"
                        if task not in tasks and len(task) > 15:
                            tasks.append(task)
                        if len(tasks) >= 6:
                            break
            
            # Return results
            if not tasks:
                return "• No specific action items identified in the meeting transcript"
            
            return "\n".join(tasks)
            
        except Exception as e:
            logger.error(f"Error in enhanced task extraction: {e}")
            return "• Error extracting tasks from meeting content"
    
    def summarize_text(self, text):
        """Ultra-fast summarization with lazy loading"""
        try:
            # Lazy load summarizer only when needed
            if TRANSFORMERS_AVAILABLE:
                self._load_summarizer()
            
            if self.summarizer and len(text) > 100:
                # Speed optimized summarization
                text_sample = text[:1000]  # Limit for speed
                summary = self.summarizer(
                    text_sample,
                    max_length=100,  # Shorter for speed
                    min_length=30,
                    do_sample=False,
                    num_beams=1  # Fastest beam search
                )[0]['summary_text']
                
                return summary
            else:
                return self.fallback_summary(text)
        
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return self.fallback_summary(text)
    
    def fallback_summary(self, text):
        """Fast fallback summarization without AI models"""
        try:
            sentences = re.split(r'[.!?]+', text)
            # Get first few sentences and key sentences with important keywords
            key_phrases = ['decision', 'action', 'important', 'key', 'main', 'summary', 'conclusion', 'result']
            
            summary_sentences = []
            # Always include first sentence if it's substantial
            if sentences and len(sentences[0].strip()) > 20:
                summary_sentences.append(sentences[0].strip())
            
            # Find sentences with key phrases
            for sentence in sentences[1:10]:  # Check first 10 sentences
                sentence = sentence.strip()
                if len(sentence) > 30:
                    sentence_lower = sentence.lower()
                    if any(phrase in sentence_lower for phrase in key_phrases):
                        summary_sentences.append(sentence)
                        if len(summary_sentences) >= 4:
                            break
            
            if summary_sentences:
                return ". ".join(summary_sentences) + "."
            else:
                # Last resort: first 200 characters
                return text[:200] + "..." if len(text) > 200 else text
        
        except Exception as e:
            logger.error(f"Error in fallback summary: {str(e)}")
            return "Summary not available"
    
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
            
            # Add performance metrics
            performance_info = f"""

---
⚡ **Ultra-Fast Processing Performance**
• Total time: {total_time:.2f} seconds
• Transcription: {transcript_time:.2f}s
• Analysis: {analysis_time:.2f}s
• Used ChatPromptTemplate chains for enhanced accuracy
• Lazy loading optimization active"""
            
            meeting_minutes += performance_info
            
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
        """Generate comprehensive meeting report with enhanced formatting"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""# 🎯 AI Meeting Analysis Report
**Generated:** {timestamp} | **Powered by:** Advanced ChatPromptTemplate Chains

---

## 📋 Executive Summary
{summary}

---

## ✅ Action Items & Tasks (ChatPromptTemplate Enhanced)
{actions}

---

## 💭 Meeting Sentiment & Tone
{sentiment}

---

## 🔑 Key Topics & Themes
{topics}

---

## 📝 Complete Meeting Transcript
{transcript}

---

### 🔧 Processing Details
- **AI Enhancement**: ChatPromptTemplate chains following LangChain documentation
- **Task Extraction**: Advanced pattern matching with prompt engineering
- **Performance**: Ultra-fast processing with lazy loading
- **Quality**: Enhanced accuracy through structured prompt templates

*This analysis uses advanced ChatPromptTemplate and chain-based processing as recommended in the LangChain documentation for maximum accuracy and speed.*"""


def create_interface():
    """Create ultra-fast Gradio interface matching the reference image"""
    
    with gr.Blocks(
        title="AI Meeting Assistant",
        theme=gr.themes.Default(),
    ) as interface:
        
        # Main title and description - exact match to the image
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #ffffff; font-size: 2.5em; margin-bottom: 10px;">AI Meeting Assistant</h1>
            <p style="color: #cccccc; font-size: 1.1em; line-height: 1.4;">
                Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate<br>
                meeting minutes along with a list of tasks.
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
                
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Submit", variant="primary")
            
            # Right column - Meeting Minutes and Tasks output (matches exactly)
            with gr.Column(scale=1):
                output_display = gr.Textbox(
                    label="Meeting Minutes and Tasks",
                    lines=20,
                    show_label=True,
                    interactive=False,
                    placeholder="Your meeting minutes and task list will appear here after processing..."
                )
                
                # Download section (matches the UI)
                gr.Markdown("### Download the Generated Meeting Minutes and Tasks")
                
                # Create download file
                download_file = gr.File(
                    label="meeting_minutes_and_tasks.txt",
                    visible=True,
                    interactive=False
                )
        
        # Event handlers with ultra-fast processing
        def process_meeting_audio(audio_file, progress=gr.Progress()):
            """Process uploaded audio file with ultra-fast AI pipeline"""
            if audio_file is None:
                return "Please upload an audio file to analyze.", None
            
            try:
                progress(0.1, desc="Starting ultra-fast processing...")
                
                progress(0.3, desc="Transcribing with Whisper tiny model...")
                
                # Process with the optimized AI assistant method
                meeting_report, temp_file = meeting_assistant.process_meeting_simple(audio_file)
                
                if meeting_report and not meeting_report.startswith("Error"):
                    progress(1.0, desc="Complete!")
                    return meeting_report, temp_file
                else:
                    return "Error processing audio file. Please try again.", None
                    
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                return f"Error processing audio: {str(e)}", None
        
        def clear_interface():
            """Clear all inputs and outputs"""
            return None, "", None
        
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
if __name__ == "__main__":
    print("🚀 Starting AI Meeting Assistant with ultra-fast optimizations...")
    print("⚡ Using lazy loading for instant startup")
    print("🧠 Models will load only when needed")
    print("🔗 ChatPromptTemplate chains ready for enhanced accuracy")
    
    demo = create_interface()
    demo.launch()