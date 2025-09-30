import gradio as gr
import os
import tempfile
from datetime import datetime
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies with graceful fallbacks
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("Whisper loaded successfully")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available - audio transcription will be disabled")

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers loaded successfully")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - using fallback methods")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain loaded successfully")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available - using simple text splitting")

class MeetingAssistant:
    def __init__(self):
        self.whisper_model = None
        self.summarizer = None
        self.sentiment_analyzer = None
        self.text_splitter = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all required models with graceful fallbacks"""
        try:
            # Initialize Whisper for speech-to-text
            if WHISPER_AVAILABLE:
                logger.info("Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            
            # Initialize Transformers models
            if TRANSFORMERS_AVAILABLE:
                logger.info("Loading AI models...")
                try:
                    # Try to load summarization model
                    self.summarizer = pipeline(
                        "summarization", 
                        model="facebook/bart-large-cnn",
                        device=-1  # Force CPU for stability
                    )
                    logger.info("Summarization model loaded")
                except Exception as e:
                    logger.warning(f"Could not load BART, trying smaller model: {e}")
                    try:
                        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                        logger.info("Fallback summarization model loaded")
                    except Exception as e2:
                        logger.warning(f"Could not load any summarization model: {e2}")
                
                try:
                    # Try to load sentiment analysis model
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=-1
                    )
                    logger.info("Sentiment analysis model loaded")
                except Exception as e:
                    logger.warning(f"Could not load RoBERTa, using default: {e}")
                    try:
                        self.sentiment_analyzer = pipeline("sentiment-analysis")
                        logger.info("Default sentiment model loaded")
                    except Exception as e2:
                        logger.warning(f"Could not load any sentiment model: {e2}")
            
            # Initialize text splitter
            if LANGCHAIN_AVAILABLE:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                logger.info("LangChain text splitter initialized")
            
            logger.info("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    def simple_text_split(self, text, chunk_size=1000):
        """Simple text splitting when LangChain is not available"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper or provide helpful message"""
        try:
            if audio_path is None:
                return "Please upload an audio file.", []
            
            if not WHISPER_AVAILABLE or self.whisper_model is None:
                return "Audio transcription is currently unavailable. Please try the Text Analysis tab to analyze meeting notes directly.", []
            
            logger.info("Starting audio transcription...")
            result = self.whisper_model.transcribe(audio_path)
            transcription = result["text"]
            
            # Extract segments with timestamps if available
            segments = []
            if "segments" in result:
                for segment in result["segments"]:
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "")
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    })
            
            logger.info("Audio transcription completed successfully")
            return transcription, segments
        
        except Exception as e:
            error_msg = f"Error transcribing audio: {str(e)}"
            logger.error(error_msg)
            return error_msg, []
    
    def simple_sentiment_analysis(self, text):
        """Simple keyword-based sentiment analysis fallback"""
        positive_words = ['good', 'great', 'excellent', 'awesome', 'fantastic', 'wonderful', 'amazing', 'perfect', 'love', 'like', 'happy', 'pleased', 'satisfied', 'success', 'successful', 'approve', 'agree', 'positive', 'benefit', 'advantage']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'frustrated', 'disappointed', 'fail', 'failure', 'problem', 'issue', 'concern', 'worried', 'disagree', 'negative', 'difficult', 'challenge', 'risk']
        
        words = re.findall(r'\b\w+\b', text.lower())
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return {"positive": 40, "negative": 20, "neutral": 40}
        
        # Calculate percentages with some baseline
        positive_percent = min(80, (positive_count / total_sentiment_words) * 70 + 15)
        negative_percent = min(80, (negative_count / total_sentiment_words) * 70 + 15)
        neutral_percent = max(5, 100 - positive_percent - negative_percent)
        
        return {
            "positive": round(positive_percent, 1),
            "negative": round(negative_percent, 1),
            "neutral": round(neutral_percent, 1)
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the text with AI or fallback method"""
        try:
            if not TRANSFORMERS_AVAILABLE or self.sentiment_analyzer is None:
                return self.simple_sentiment_analysis(text)
            
            # Split text into manageable chunks
            if self.text_splitter:
                chunks = self.text_splitter.split_text(text)
            else:
                chunks = self.simple_text_split(text, 512)
            
            sentiments = []
            for chunk in chunks[:5]:  # Limit to first 5 chunks for performance
                if len(chunk.strip()) > 10:
                    try:
                        result = self.sentiment_analyzer(chunk)
                        sentiments.append(result[0])
                    except Exception as e:
                        logger.warning(f"Error analyzing chunk sentiment: {e}")
                        continue
            
            if not sentiments:
                return self.simple_sentiment_analysis(text)
            
            # Map different label formats to standard format
            positive_labels = ['POSITIVE', 'LABEL_2']
            negative_labels = ['NEGATIVE', 'LABEL_0'] 
            neutral_labels = ['NEUTRAL', 'LABEL_1']
            
            positive_count = sum(1 for s in sentiments if s['label'] in positive_labels)
            negative_count = sum(1 for s in sentiments if s['label'] in negative_labels)
            neutral_count = sum(1 for s in sentiments if s['label'] in neutral_labels)
            
            total = len(sentiments)
            return {
                "positive": round((positive_count / total) * 100, 1) if total > 0 else 33,
                "negative": round((negative_count / total) * 100, 1) if total > 0 else 33,
                "neutral": round((neutral_count / total) * 100, 1) if total > 0 else 34
            }
        
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return self.simple_sentiment_analysis(text)
    
    def create_simple_summary(self, text, summary_type):
        """Create a simple summary when AI summarization is not available"""
        # Get the most important sentences (first few and any with key indicators)
        sentences = re.split(r'[.!?]+', text)
        important_sentences = []
        
        # Take first few sentences
        for i, sentence in enumerate(sentences[:6]):
            sentence = sentence.strip()
            if len(sentence) > 20:
                important_sentences.append(sentence)
        
        # Look for sentences with important keywords
        key_phrases = ['decision', 'action', 'next step', 'important', 'key', 'main', 'summary', 'conclusion', 'result']
        for sentence in sentences[6:20]:  # Check next 14 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:
                sentence_lower = sentence.lower()
                if any(phrase in sentence_lower for phrase in key_phrases):
                    if sentence not in important_sentences:
                        important_sentences.append(sentence)
                        if len(important_sentences) >= 8:
                            break
        
        if not important_sentences:
            important_sentences = [sentences[0].strip()] if sentences and sentences[0].strip() else ["No content available for summary."]
        
        summary = '. '.join(important_sentences)
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return self.format_summary(summary, summary_type)
    
    def format_summary(self, summary, summary_type):
        """Format summary based on requested type"""
        if not summary.strip():
            return "Unable to generate summary from the provided content."
            
        if summary_type == "bullet_points":
            sentences = re.split(r'[.!?]+', summary)
            bullet_points = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15:
                    bullet_points.append(f"• {sentence}")
            return "\n".join(bullet_points[:8]) if bullet_points else "• No key points identified"
        
        elif summary_type == "executive_summary":
            return f"**Executive Summary:**\n\n{summary}"
        
        else:  # paragraph
            return summary
    
    def summarize_text(self, text, summary_type="bullet_points"):
        """Summarize the text using AI or fallback method"""
        try:
            if len(text.strip()) < 100:
                return "Text is too short for meaningful summarization. Please provide more content."
            
            if not TRANSFORMERS_AVAILABLE or self.summarizer is None:
                return self.create_simple_summary(text, summary_type)
            
            # Split text into manageable chunks
            if self.text_splitter:
                chunks = self.text_splitter.split_text(text)
            else:
                chunks = self.simple_text_split(text, 1000)
            
            summaries = []
            for chunk in chunks[:3]:  # Process first 3 chunks for performance
                if len(chunk.strip()) > 100:
                    try:
                        summary_result = self.summarizer(
                            chunk, 
                            max_length=130, 
                            min_length=30, 
                            do_sample=False
                        )
                        summaries.append(summary_result[0]['summary_text'])
                    except Exception as e:
                        logger.warning(f"Error summarizing chunk: {e}")
                        continue
            
            if not summaries:
                return self.create_simple_summary(text, summary_type)
            
            # Combine all summaries
            combined_summary = " ".join(summaries)
            return self.format_summary(combined_summary, summary_type)
        
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return self.create_simple_summary(text, summary_type)
    
    def extract_action_items(self, text):
        """Extract action items and tasks from the text"""
        try:
            action_patterns = [
                r'\b(action item|follow up|next step|to do|todo|task)\b.*?[.!?]',
                r'\b(assign|responsible|deadline|complete|finish)\b.*?[.!?]',
                r'\b(schedule|plan|implement|review|check|prepare)\b.*?[.!?]',
                r'\b(need to|should|must|will|going to)\b.*?\b(by|before|until|next week|tomorrow|monday|tuesday|wednesday|thursday|friday)\b.*?[.!?]',
                r'\b(send|email|call|contact|reach out|create|make|build|update|inform|notify)\b.*?[.!?]'
            ]
            
            action_items = []
            for pattern in action_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(match)
                    
                    # Clean up the match
                    match = re.sub(r'\s+', ' ', match).strip()
                    if len(match) > 20 and len(match) < 200:
                        # Capitalize first letter
                        match = match[0].upper() + match[1:] if match else ""
                        formatted_item = f"• {match}"
                        if formatted_item not in action_items:
                            action_items.append(formatted_item)
                        
                        if len(action_items) >= 10:
                            break
                
                if len(action_items) >= 10:
                    break
            
            # If no explicit action items found, look for imperative sentences
            if len(action_items) < 3:
                sentences = re.split(r'[.!?]', text)
                imperative_indicators = ['please', 'make sure', 'ensure', 'remember to', 'don\'t forget']
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20 and len(sentence) < 200:
                        sentence_lower = sentence.lower()
                        if any(indicator in sentence_lower for indicator in imperative_indicators):
                            formatted_item = f"• {sentence.capitalize()}"
                            if formatted_item not in action_items:
                                action_items.append(formatted_item)
                                if len(action_items) >= 8:
                                    break
            
            if not action_items:
                action_items = ["• No clear action items identified in the text"]
            
            return "\n".join(action_items[:10])
        
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
            return "• Error analyzing text for action items"
    
    def identify_key_topics(self, text):
        """Identify key topics and important themes"""
        try:
            # Extract meaningful words (4+ characters, not common stop words)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            
            # Comprehensive stop words list
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with', 'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'would', 'there', 'could', 'other', 'after', 'first', 'never', 'these', 'think', 'where', 'being', 'every', 'great', 'might', 'shall', 'still', 'those', 'under', 'while', 'again', 'before', 'right', 'about', 'also', 'back', 'call', 'came', 'each', 'even', 'going', 'look', 'most', 'move', 'need', 'only', 'said', 'same', 'show', 'tell', 'turn', 'ways', 'went', 'work', 'year', 'yes'
            }
            
            # Count word frequencies
            word_count = {}
            for word in words:
                if word not in stop_words and len(word) > 3:
                    word_count[word] = word_count.get(word, 0) + 1
            
            if not word_count:
                return "• No significant topics identified"
            
            # Sort by frequency and get top words
            top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:15]
            
            topics = []
            for word, count in top_words:
                if count > 1:  # Only include words mentioned multiple times
                    topics.append(f"• {word.capitalize()} (mentioned {count} times)")
            
            if not topics:
                # If no repeated words, just take the most common single mentions
                for word, count in top_words[:8]:
                    topics.append(f"• {word.capitalize()}")
            
            return "\n".join(topics) if topics else "• No significant topics identified"
        
        except Exception as e:
            logger.error(f"Error identifying topics: {str(e)}")
            return "• Error analyzing topics from text"
    
    def process_meeting_text(self, input_text, summary_type="bullet_points"):
        """Process meeting text and generate comprehensive analysis"""
        try:
            if not input_text or len(input_text.strip()) < 50:
                return "Please provide meeting text with at least 50 characters for analysis.", "", "", "", "", ""
            
            transcription = input_text.strip()
            logger.info("Processing meeting text...")
            
            # Generate summary
            summary = self.summarize_text(transcription, summary_type)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(transcription)
            sentiment_text = f"""**Meeting Sentiment Analysis:**
• Positive: {sentiment['positive']}%
• Neutral: {sentiment['neutral']}%  
• Negative: {sentiment['negative']}%"""
            
            # Extract action items
            action_items = self.extract_action_items(transcription)
            
            # Identify key topics
            topics = self.identify_key_topics(transcription)
            
            # Create comprehensive meeting report
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meeting_report = f"""# 🎯 Meeting Analysis Report
**Generated on:** {timestamp}

## 📋 Meeting Summary
{summary}

## 📊 Sentiment Analysis
{sentiment_text}

## ✅ Action Items
{action_items}

## 🔑 Key Topics Discussed
{topics}

## 📝 Full Transcription
{transcription}

---
*Generated by AI Meeting Assistant*"""
            
            logger.info("Meeting text processing completed successfully")
            return transcription, summary, sentiment_text, action_items, topics, meeting_report
        
        except Exception as e:
            error_msg = f"Error processing text: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", "", "", "", ""
    
    def process_meeting_audio(self, audio_file, summary_type="bullet_points"):
        """Process meeting audio file and generate analysis"""
        try:
            if audio_file is None:
                return "Please upload an audio file or use the Text Analysis tab to analyze meeting notes directly.", "", "", "", "", ""
            
            logger.info("Starting audio processing...")
            
            # Transcribe audio
            transcription, segments = self.transcribe_audio(audio_file)
            
            if transcription.startswith("Error") or transcription.startswith("Please") or transcription.startswith("Audio transcription"):
                return transcription, "", "", "", "", ""
            
            # Process the transcribed text
            return self.process_meeting_text(transcription, summary_type)
        
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", "", "", "", ""

# Initialize the meeting assistant
meeting_assistant = MeetingAssistant()

def create_interface():
    """Create the Gradio interface matching the exact UI from the reference image"""
    
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
                
                # Status display for processing feedback
                status_display = gr.HTML(visible=False)
            
            # Right column - Meeting Minutes and Tasks output (matches exactly)
            with gr.Column(scale=1):
                output_display = gr.Textbox(
                    label="Meeting Minutes and Tasks",
                    lines=20,
                    show_label=True,
                    interactive=False,
                    placeholder="Your meeting minutes and task list will appear here after processing..."
                )
                
                # Download section with workflow image on the left
                with gr.Row():
                    # Left side: Workflow image (transparent) - Matching original attached image
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="display: flex; justify-content: center; align-items: center; height: 300px; padding: 20px;">
                            <svg width="100%" height="100%" viewBox="0 0 1000 500" style="max-width: 100%; max-height: 100%; opacity: 0.75;">
                                <!-- Title -->
                                <text x="500" y="40" text-anchor="middle" fill="#ffffff" font-size="28" font-weight="bold">AI Meeting Assistant Workflow</text>
                                
                                <!-- Meeting Recording -->
                                <rect x="50" y="80" width="120" height="90" rx="12" fill="#667eea" opacity="0.9"/>
                                <text x="110" y="115" text-anchor="middle" fill="white" font-size="32">🎤</text>
                                <text x="110" y="140" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Meeting</text>
                                <text x="110" y="158" text-anchor="middle" fill="white" font-size="12">Recording</text>
                                
                                <!-- Arrow -->
                                <path d="M180 125 L220 125 M212 118 L220 125 L212 132" stroke="#4ecdc4" stroke-width="4" fill="none"/>
                                
                                <!-- Transcribe -->
                                <rect x="230" y="80" width="120" height="90" rx="12" fill="#f093fb" opacity="0.9"/>
                                <text x="290" y="115" text-anchor="middle" fill="white" font-size="28">🧠</text>
                                <text x="290" y="135" text-anchor="middle" fill="white" font-size="10" font-weight="bold">OpenAI Whisper</text>
                                <text x="290" y="150" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Transcribe</text>
                                
                                <!-- Arrow -->
                                <path d="M360 125 L400 125 M392 118 L400 125 L392 132" stroke="#4ecdc4" stroke-width="4" fill="none"/>
                                
                                <!-- Clean-Up -->
                                <rect x="410" y="80" width="120" height="90" rx="12" fill="#667eea" opacity="0.9"/>
                                <text x="470" y="115" text-anchor="middle" fill="white" font-size="28">👤</text>
                                <text x="470" y="135" text-anchor="middle" fill="white" font-size="10" font-weight="bold">LLAMA 3.2</text>
                                <text x="470" y="150" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Clean-Up</text>
                                
                                <!-- Arrow -->
                                <path d="M540 125 L580 125 M572 118 L580 125 L572 132" stroke="#4ecdc4" stroke-width="4" fill="none"/>
                                
                                <!-- Minutes Generator -->
                                <rect x="590" y="80" width="120" height="90" rx="12" fill="#667eea" opacity="0.9"/>
                                <text x="650" y="115" text-anchor="middle" fill="white" font-size="28">👤</text>
                                <text x="650" y="135" text-anchor="middle" fill="white" font-size="10" font-weight="bold">Granite 3.0</text>
                                <text x="650" y="150" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Minutes</text>
                                
                                <!-- Arrow -->
                                <path d="M720 125 L760 125 M752 118 L760 125 L752 132" stroke="#4ecdc4" stroke-width="4" fill="none"/>
                                
                                <!-- Gradio Interface -->
                                <rect x="770" y="80" width="120" height="90" rx="12" fill="#667eea" opacity="0.9"/>
                                <text x="830" y="115" text-anchor="middle" fill="white" font-size="32">💻</text>
                                <text x="830" y="140" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Gradio</text>
                                <text x="830" y="158" text-anchor="middle" fill="white" font-size="12">Interface</text>
                                
                                <!-- Supporting Tools Box -->
                                <rect x="350" y="220" width="300" height="80" rx="12" fill="#4a4a4a" opacity="0.8" stroke="#4ecdc4" stroke-width="2"/>
                                <text x="500" y="245" text-anchor="middle" fill="#4ecdc4" font-size="16" font-weight="bold">🛠️ Supporting Technologies</text>
                                
                                <!-- Prompt Template -->
                                <rect x="380" y="260" width="100" height="30" rx="6" fill="#5a5a5a" opacity="0.7"/>
                                <text x="405" y="275" fill="white" font-size="18">📄</text>
                                <text x="430" y="280" fill="#cccccc" font-size="12" font-weight="bold">Prompt</text>
                                <text x="430" y="290" fill="#cccccc" font-size="10">Template</text>
                                
                                <!-- LangChain -->
                                <rect x="520" y="260" width="100" height="30" rx="6" fill="#5a5a5a" opacity="0.7"/>
                                <text x="545" y="275" fill="white" font-size="18">🔗</text>
                                <text x="570" y="280" fill="#cccccc" font-size="12" font-weight="bold">LangChain</text>
                                <text x="570" y="290" fill="#cccccc" font-size="10">Processing</text>
                                
                                <!-- Arrow from Clean-Up to Supporting Tools -->
                                <path d="M470 180 Q470 200 450 220" stroke="#4ecdc4" stroke-width="3" fill="none" opacity="0.7"/>
                                
                                <!-- Arrow from Supporting Tools to Minutes Generator -->
                                <path d="M550 220 Q650 200 650 180" stroke="#4ecdc4" stroke-width="3" fill="none" opacity="0.7"/>
                                
                                <!-- Additional Labels -->
                                <text x="500" y="380" text-anchor="middle" fill="#cccccc" font-size="14" font-style="italic">Complete AI-Powered Meeting Processing Pipeline</text>
                                <text x="500" y="420" text-anchor="middle" fill="#888888" font-size="12">Speech-to-Text → Text Processing → Intelligent Analysis → Structured Output</text>
                            </svg>
                        </div>
                        """)
                    
                    # Right side: Download checkbox and file
                    with gr.Column(scale=2):
                        gr.HTML("""
                        <div style="margin-top: 15px;">
                            <label style="display: flex; align-items: center; cursor: pointer; color: #cccccc;">
                                <input type="checkbox" id="download-checkbox" style="margin-right: 8px;" checked> 
                                Download the Generated Meeting Minutes and Tasks
                            </label>
                        </div>
                        """)
                        
                        # Create download file
                        download_file = gr.File(
                            label="meeting_minutes_and_tasks.txt",
                            visible=True,
                            interactive=False
                        )

        
        # Event handlers with full AI functionality
        def process_meeting_audio(audio_file, progress=gr.Progress()):
            """Process uploaded audio file with full AI pipeline"""
            if audio_file is None:
                return "Please upload an audio file to analyze.", None, """<div style="color: #ff6b6b; padding: 10px; background-color: #2a2a2a; border-radius: 5px;">
                ❌ No audio file provided. Please upload an audio file to continue.
                </div>"""
            
            try:
                progress(0.1, desc="Starting audio processing...")
                
                # Update status
                status_html = """<div style="color: #4ecdc4; padding: 10px; background-color: #2a2a2a; border-radius: 5px;">
                🎵 Processing audio file... This may take a few minutes for the first run.
                </div>"""
                
                progress(0.3, desc="Transcribing audio...")
                
                # Process with the AI assistant - use the correct method
                results = meeting_assistant.process_meeting_audio(audio_file, "bullet_points")
                
                if len(results) >= 6:
                    transcription, summary, sentiment, actions, topics, report = results
                    
                    progress(0.8, desc="Generating meeting minutes...")
                    
                    # Format output exactly like the reference image
                    meeting_minutes = f"""Meeting Minutes:
{summary}

Task List:
{actions}"""
                    
                    progress(0.9, desc="Creating download file...")
                    
                    # Create downloadable file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                        f.write(f"""AI Meeting Assistant - Generated Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{report}""")
                        temp_file_path = f.name
                    
                    progress(1.0, desc="Complete!")
                    
                    success_status = """<div style="color: #4ecdc4; padding: 10px; background-color: #2a2a2a; border-radius: 5px;">
                    ✅ Meeting analysis complete! Minutes and tasks generated successfully.
                    </div>"""
                    
                    return meeting_minutes, temp_file_path, success_status
                else:
                    error_status = """<div style="color: #ff6b6b; padding: 10px; background-color: #2a2a2a; border-radius: 5px;">
                    ❌ Error processing audio. Please try again with a different file.
                    </div>"""
                    return "Error processing audio file. Please try again.", None, error_status
                    
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                error_status = f"""<div style="color: #ff6b6b; padding: 10px; background-color: #2a2a2a; border-radius: 5px;">
                ❌ Error: {str(e)}
                </div>"""
                return f"Error processing audio: {str(e)}", None, error_status
        
        def clear_interface():
            """Clear all inputs and outputs"""
            return None, "", None, """<div style="color: #cccccc; padding: 10px; background-color: #2a2a2a; border-radius: 5px;">
            Interface cleared. Upload an audio file to begin analysis.
            </div>"""
        

        
        # Wire up the event handlers
        submit_btn.click(
            fn=process_meeting_audio,
            inputs=[audio_input],
            outputs=[output_display, download_file, status_display]
        )
        
        clear_btn.click(
            fn=clear_interface,
            inputs=[],
            outputs=[audio_input, output_display, download_file, status_display]
        )
        
        # Footer with comprehensive information - Dark theme compatible
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #1a1a1a; border-radius: 8px; border: 1px solid #444;">
            <h3 style="color: #ffffff;">🚀 About This AI Meeting Assistant</h3>
            <p style="color: #cccccc;">Comprehensive meeting analysis with intelligent fallback systems for maximum reliability:</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
                <div style="background: #2a4a5a; padding: 15px; border-radius: 8px; border: 1px solid #3a5a6a;">
                    <strong style="color: #ffffff;">🎙️ Audio Processing</strong><br>
                    <span style="color: #cccccc;">Whisper AI for speech-to-text conversion</span>
                </div>
                <div style="background: #4a2a5a; padding: 15px; border-radius: 8px; border: 1px solid #5a3a6a;">
                    <strong style="color: #ffffff;">📝 Text Analysis</strong><br>
                    <span style="color: #cccccc;">Direct text processing and analysis</span>
                </div>
                <div style="background: #2a5a2a; padding: 15px; border-radius: 8px; border: 1px solid #3a6a3a;">
                    <strong style="color: #ffffff;">🧠 AI Models</strong><br>
                    <span style="color: #cccccc;">BART, RoBERTa + intelligent fallbacks</span>
                </div>
                <div style="background: #5a4a2a; padding: 15px; border-radius: 8px; border: 1px solid #6a5a3a;">
                    <strong style="color: #ffffff;">📊 Insights</strong><br>
                    <span style="color: #cccccc;">Summary, Sentiment, Actions, Topics</span>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <h4 style="color: #ffffff;">🎯 Perfect for:</h4>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin: 15px 0;">
                    <span style="color: #cccccc;">• Business Meetings</span>
                    <span style="color: #cccccc;">• Interviews</span>
                    <span style="color: #cccccc;">• Lectures</span>
                    <span style="color: #cccccc;">• Brainstorming</span>
                    <span style="color: #cccccc;">• Conference Calls</span>
                </div>
            </div>
            <p style="margin-top: 20px; font-size: 0.9em; color: #aaaaaa;">
                Built with ❤️ using Gradio, Transformers, LangChain, and Whisper<br>
                Open source • Free to use • Privacy-focused
            </p>
            <p style="font-size: 0.8em; color: #888888; margin-top: 10px;">
                <strong>Reliability Note:</strong> This app includes intelligent fallback methods to ensure functionality 
                even when advanced AI models are unavailable, guaranteeing useful analysis in all scenarios.
            </p>
        </div>
        """)
    
    return interface

# Launch the application
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )