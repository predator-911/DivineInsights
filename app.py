import streamlit as st
import random
import re
import json
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import time
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# --- Page config ---
st.set_page_config(
    page_title="Divine Insights - Hindu Scriptures Explained",
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🕉️"
)

# --- App State and Session Variables ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Home"
if 'user_comments' not in st.session_state:
    st.session_state.user_comments = {}
if 'reading_progress' not in st.session_state:
    st.session_state.reading_progress = {}
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"

# --- Load Hugging Face pipelines once ---
@st.cache_resource(show_spinner=False)
def load_pipelines():
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        sentiment = pipeline("sentiment-analysis")
        zero_shot = pipeline("zero-shot-classification")
        return summarizer, sentiment, zero_shot, True
    except (ImportError, RuntimeError):
        st.warning("AI features limited. Install transformers package for full functionality.")
        return None, None, None, False

summarizer, sentiment_analyzer, zero_shot_classifier, models_available = load_pipelines()

# --- Load sentence transformer if available ---
@st.cache_resource
def load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model, True
    except (ImportError, RuntimeError):
        return None, False

sentence_model, sentence_model_available = load_sentence_transformer()

# --- Theme Configuration ---
def set_theme(theme_name):
    if theme_name == "Light":
        # Default light theme
        return
    elif theme_name == "Dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    elif theme_name == "Krishna":
        st.markdown("""
        <style>
        .stApp {
            background-color: #264653;
            color: #f1faee;
        }
        .stButton>button {
            background-color: #e76f51;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    elif theme_name == "Shiva":
        st.markdown("""
        <style>
        .stApp {
            background-color: #212529;
            color: #e9ecef;
        }
        .stButton>button {
            background-color: #6c757d;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# --- Data: Extended Scripture Database ---
@st.cache_data
def load_scripture_data():
    # This would ideally be loaded from JSON files or a database
    scriptures = {
        "Bhagavad Gita": {
            "Chapter 2, Verse 47": {
                "sanskrit": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।\nमा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
                "translation": "You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions.",
                "explanation": "This verse teaches the principle of detached action. One should focus on performing their duty without being attached to the results. This is the essence of Karma Yoga.",
                "themes": ["karma yoga", "duty", "detachment"],
                "related_verses": ["Chapter 3, Verse 30", "Chapter 18, Verse 48"],
                "difficulty": "Beginner",
                "audio_url": None  # Would contain URL to audio file if available
            },
            "Chapter 12, Verse 15": {
                "sanskrit": "यस्मान्नोद्विजते लोको लोकान्नोद्विजते च य: ।\nहर्षामर्षभयोद्वेगैर्मुक्तो य: स च मे प्रिय: ॥",
                "translation": "He by whom the world is not agitated and who cannot be agitated by the world, who is freed from joy, envy, fear, and anxiety-he is dear to Me.",
                "explanation": "This verse describes the qualities of a devotee who is dear to Krishna. Such a person remains equanimous and does not disturb others or get disturbed by the world.",
                "themes": ["equanimity", "devotion", "emotional stability"],
                "related_verses": ["Chapter 12, Verse 13", "Chapter 12, Verse 16"],
                "difficulty": "Intermediate",
                "audio_url": None
            },
            "Chapter 3, Verse 30": {
                "sanskrit": "मयि सर्वाणि कर्माणि संन्यस्याध्यात्मचेतसा।\nनिराशीर्निर्ममो भूत्वा युध्यस्व विगतज्वरः॥",
                "translation": "Dedicating all actions to Me, with your mind fixed on Me, free from desire and selfishness, fight without mental fever.",
                "explanation": "Krishna instructs Arjuna to dedicate all actions to the Divine while maintaining mental focus, detachment from results, and freedom from ego and anxiety.",
                "themes": ["surrender", "duty", "detachment", "action"],
                "related_verses": ["Chapter 2, Verse 47", "Chapter 5, Verse 10"],
                "difficulty": "Intermediate",
                "audio_url": None
            },
            "Chapter 18, Verse 48": {
                "sanskrit": "सहजं कर्म कौन्तेय सदोषमपि न त्यजेत्।\nसर्वारम्भा हि दोषेण धूमेनाग्निरिवावृताः॥",
                "translation": "One should not abandon duties born of one's nature, even if they are flawed, for all undertakings are covered by defects as fire is covered by smoke.",
                "explanation": "This verse emphasizes that one should perform their natural duties (sva-dharma) despite imperfections, as all actions in the material world have some flaws.",
                "themes": ["sva-dharma", "duty", "imperfection"],
                "related_verses": ["Chapter 2, Verse 47", "Chapter 3, Verse 35"],
                "difficulty": "Advanced",
                "audio_url": None
            }
        },
        "Upanishads": {
            "Isha Upanishad, Verse 1": {
                "sanskrit": "ईशा वास्यमिदं सर्वं यत्किञ्च जगत्यां जगत्।\nतेन त्यक्तेन भुञ्जीथा मा गृधः कस्यस्विद्धनम्॥",
                "translation": "All this, whatever exists in this changing universe, should be covered by the Lord. Protect yourself through that detachment. Do not covet anybody's wealth.",
                "explanation": "This verse teaches that everything in the universe is permeated by the Divine. One should enjoy life with detachment, recognizing the Divine presence everywhere.",
                "themes": ["divinity", "detachment", "renunciation"],
                "related_verses": ["Bhagavad Gita, Chapter 5, Verse 18"],
                "difficulty": "Intermediate",
                "audio_url": None
            },
            "Katha Upanishad, 1.3.14": {
                "sanskrit": "उत्तिष्ठत जाग्रत प्राप्य वरान्निबोधत।\nक्षुरस्य धारा निशिता दुरत्यया दुर्गं पथस्तत्कवयो वदन्ति॥",
                "translation": "Arise, awake, and learn by approaching the exalted ones, for that path is difficult to traverse as the sharp edge of a razor.",
                "explanation": "This verse is a call to spiritual awakening. It emphasizes the importance of seeking knowledge from realized masters, as the spiritual path is challenging.",
                "themes": ["spiritual awakening", "knowledge", "guidance"],
                "related_verses": ["Bhagavad Gita, Chapter 4, Verse 34"],
                "difficulty": "Advanced",
                "audio_url": None
            }
        },
        "Yoga Sutras": {
            "Chapter 1, Sutra 2": {
                "sanskrit": "योगश्चित्तवृत्तिनिरोधः",
                "translation": "Yoga is the cessation of the modifications of the mind.",
                "explanation": "This foundational sutra defines yoga as the practice of stilling the fluctuations or changes in consciousness, resulting in a state of pure awareness.",
                "themes": ["meditation", "mind control", "awareness"],
                "related_verses": ["Bhagavad Gita, Chapter 6, Verse 20"],
                "difficulty": "Beginner",
                "audio_url": None
            },
            "Chapter 2, Sutra 46": {
                "sanskrit": "स्थिरसुखमासनम्",
                "translation": "Posture (asana) should be steady and comfortable.",
                "explanation": "This sutra describes the ideal qualities of yoga postures: steadiness and comfort. It applies to both physical postures and meditation.",
                "themes": ["asana", "comfort", "stability"],
                "related_verses": ["Bhagavad Gita, Chapter 6, Verse 11"],
                "difficulty": "Beginner",
                "audio_url": None
            }
        }
    }
    return scriptures

scriptures = load_scripture_data()

# --- Helper functions ---
def get_all_verses():
    all_verses = []
    for scripture, verses in scriptures.items():
        for verse_key, verse_data in verses.items():
            all_verses.append({
                "scripture": scripture,
                "verse_key": verse_key,
                "translation": verse_data["translation"],
                "themes": verse_data.get("themes", []),
                "difficulty": verse_data.get("difficulty", "Intermediate")
            })
    return all_verses

def chunk_text(text: str, max_tokens: int = 250) -> List[str]:
    # Simple chunking by sentences approx to max_tokens words
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk.split()) + len(sent.split()) <= max_tokens:
            current_chunk += " " + sent if current_chunk else sent
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def show_search_page():  # Add this function
    st.title("🔍 Search & Discover")
    
    search_term = st.text_input("Enter keyword or phrase:")
    if search_term:
        results = find_verses_by_keyword(search_term)
        
        if results:
            st.subheader(f"Found {len(results)} matches")
            for verse in results:
                with st.expander(f"{verse['scripture']} - {verse['verse_key']}"):
                    st.markdown(f"**Translation**: {verse['text']}")
                    st.markdown(f"**Themes**: {', '.join(verse['themes'])}")
        else:
            st.warning("No verses found matching your search")
            
def find_verses_by_keyword(keyword):
    matching_verses = []
    for scripture, verses in scriptures.items():
        for verse_key, verse_data in verses.items():
            # Search in translation and themes
            if (keyword.lower() in verse_data["translation"].lower() or 
                any(keyword.lower() in theme.lower() for theme in verse_data.get("themes", []))):
                matching_verses.append({
                    "scripture": scripture,
                    "verse_key": verse_key,
                    "text": verse_data["translation"],
                    "themes": verse_data.get("themes", [])
                })
    return matching_verses

def find_similar_verses(scripture, verse_key):
    if not sentence_model_available:
        # Fallback to pre-defined related verses
        related = scriptures[scripture][verse_key].get("related_verses", [])
        results = []
        for related_verse in related:
            # Parse the related verse reference
            if "," in related_verse:
                related_scripture = related_verse.split(",")[0].strip()
                related_verse_key = related_verse
                if related_scripture in scriptures:
                    for v_key, v_data in scriptures[related_scripture].items():
                        if related_verse in v_key:
                            results.append({
                                "scripture": related_scripture,
                                "verse_key": v_key,
                                "text": v_data["translation"],
                                "similarity": 0.9  # Placeholder similarity score
                            })
            else:
                # Handle case where it's in the same scripture
                for v_key, v_data in scriptures[scripture].items():
                    if related_verse in v_key:
                        results.append({
                            "scripture": scripture,
                            "verse_key": v_key,
                            "text": v_data["translation"],
                            "similarity": 0.9  # Placeholder similarity score
                        })
        return results
    else:
        # Use sentence transformers for similarity
        target_embedding = sentence_model.encode(scriptures[scripture][verse_key]["translation"])
        results = []
        
        for s_name, s_verses in scriptures.items():
            for v_key, v_data in s_verses.items():
                # Skip the same verse
                if s_name == scripture and v_key == verse_key:
                    continue
                
                verse_embedding = sentence_model.encode(v_data["translation"])
                similarity = float(np.dot(target_embedding, verse_embedding) / 
                                 (np.linalg.norm(target_embedding) * np.linalg.norm(verse_embedding)))
                
                if similarity > 0.5:  # Threshold for similarity
                    results.append({
                        "scripture": s_name,
                        "verse_key": v_key,
                        "text": v_data["translation"],
                        "similarity": similarity
                    })
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]  # Return top 5 similar verses

def categorize_text(text):
    if not models_available or zero_shot_classifier is None:
        # Fallback to basic keyword matching
        themes = []
        theme_keywords = {
            "devotion": ["devotion", "devotee", "worship", "love", "bhakti"],
            "knowledge": ["knowledge", "wisdom", "understand", "intellect", "jnana"],
            "duty": ["duty", "action", "work", "karma", "responsibility"],
            "meditation": ["meditation", "concentrate", "focus", "mind", "dhyana"],
            "detachment": ["detachment", "renunciation", "unattached", "desire", "free"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                themes.append(theme)
                
        return themes if themes else ["general spiritual wisdom"]
    else:
        # Use zero-shot classification for themes
        candidate_labels = [
            "devotion", "knowledge", "duty", "meditation", "detachment",
            "karma yoga", "bhakti yoga", "jnana yoga", "raja yoga",
            "ethics", "self-realization", "enlightenment"
        ]
        
        result = zero_shot_classifier(text, candidate_labels)
        # Filter for scores above threshold
        themes = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.5]
        return themes if themes else ["general spiritual wisdom"]

def summarize_text(text: str) -> str:
    if not models_available or summarizer is None:
        # Simple extractive summarization as fallback
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) <= 2:
            return text
        return " ".join(sentences[:2])  # Return first two sentences as summary
    
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
        summaries.append(summary)
    # Combine summaries and optionally summarize again if too long
    combined_summary = " ".join(summaries)
    if len(combined_summary.split()) > 100:
        combined_summary = summarizer(combined_summary, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]
    return combined_summary

def analyze_sentiment(text: str):
    if not models_available or sentiment_analyzer is None:
        # Simple rule-based sentiment fallback
        positive_words = ["love", "joy", "peace", "harmony", "divine", "good", "enlighten", "happiness"]
        negative_words = ["fear", "anxiety", "suffering", "pain", "ignorance", "evil", "bondage"]
        
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        
        if pos_count > neg_count:
            return "POSITIVE", 0.7, []
        elif neg_count > pos_count:
            return "NEGATIVE", 0.7, []
        else:
            return "NEUTRAL", 0.5, []
    
    chunks = chunk_text(text, max_tokens=128)
    results = []
    for chunk in chunks:
        res = sentiment_analyzer(chunk)[0]
        results.append(res)
    # Aggregate sentiment counts and average confidence
    label_counts = {}
    total_score = 0
    for r in results:
        label = r['label']
        label_counts[label] = label_counts.get(label, 0) + 1
        total_score += r['score']
    avg_score = total_score / len(results) if results else 0
    # Determine dominant sentiment
    dominant_sentiment = max(label_counts, key=label_counts.get) if label_counts else "NEUTRAL"
    return dominant_sentiment, avg_score, results

def generate_wordcloud(text):
    # Generate a word cloud image from text
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         contour_width=3, contour_color='steelblue').generate(text)
    
    # Create a figure for the word cloud with specified size
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Convert the figure to a PNG image
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert PNG to base64 for displaying in HTML
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)  # Close the figure to free memory
    
    return img_str

def mark_verse_as_read(scripture, verse_key):
    key = f"{scripture}|{verse_key}"
    st.session_state.reading_progress[key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success(f"Marked '{verse_key}' from {scripture} as read!")

def add_to_favorites(scripture, verse_key):
    favorite = {
        "scripture": scripture,
        "verse_key": verse_key,
        "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Check if already in favorites
    if not any(f["scripture"] == scripture and f["verse_key"] == verse_key for f in st.session_state.favorites):
        st.session_state.favorites.append(favorite)
        st.success(f"Added '{verse_key}' from {scripture} to favorites!")
    else:
        st.info(f"'{verse_key}' from {scripture} is already in your favorites.")

def add_comment(scripture, verse_key, comment):
    key = f"{scripture}|{verse_key}"
    if key not in st.session_state.user_comments:
        st.session_state.user_comments[key] = []
    
    st.session_state.user_comments[key].append({
        "text": comment,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username": "User"  # Could be customized if user authentication is added
    })
    
    st.success("Comment added successfully!")

def get_comments(scripture, verse_key):
    key = f"{scripture}|{verse_key}"
    return st.session_state.user_comments.get(key, [])

def is_verse_read(scripture, verse_key):
    key = f"{scripture}|{verse_key}"
    return key in st.session_state.reading_progress

def get_reading_stats():
    stats = {}
    for scripture in scriptures:
        total_verses = len(scriptures[scripture])
        read_verses = sum(1 for key in st.session_state.reading_progress if key.startswith(f"{scripture}|"))
        stats[scripture] = {
            "total": total_verses,
            "read": read_verses,
            "percentage": (read_verses / total_verses * 100) if total_verses > 0 else 0
        }
    return stats

def export_verse_to_image(scripture, verse_key, verse_data):
    # Create a figure with white background
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.axis('off')
    
    # Add text with proper formatting
    plt.text(0.5, 0.95, f"{scripture} - {verse_key}", fontsize=16, 
             horizontalalignment='center', verticalalignment='top', fontweight='bold')
    
    plt.text(0.5, 0.85, verse_data["sanskrit"], fontsize=14,
             horizontalalignment='center', verticalalignment='top', style='italic')
    
    plt.text(0.5, 0.7, verse_data["translation"], fontsize=12,
             horizontalalignment='center', verticalalignment='top', wrap=True)
    
    if "explanation" in verse_data:
        plt.text(0.5, 0.5, verse_data["explanation"], fontsize=10,
                 horizontalalignment='center', verticalalignment='top', wrap=True)
    
    plt.text(0.5, 0.1, "Generated with Divine Insights App", fontsize=8,
             horizontalalignment='center', verticalalignment='bottom', color='gray')
    
    # Save to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# --- Main Application UI ---
def main():
    # Apply selected theme
    set_theme(st.session_state.theme)
    
    # Sidebar for navigation and settings
    with st.sidebar:
        # Create local "Om" symbol using Unicode character instead of loading from URL
        st.markdown("<h1 style='text-align: center; font-size: 48px;'>🕉️</h1>", unsafe_allow_html=True)
        st.title("Divine Insights")
        
        # Navigation
        nav_options = ["Home", "Scripture Explorer", "Search & Discover", "Reading Progress", "Personal Notes", "Meditation Timer"]
        selected_nav = st.radio("Navigation", nav_options, index=nav_options.index(st.session_state.current_tab))
        
        if selected_nav != st.session_state.current_tab:
            st.session_state.current_tab = selected_nav
            st.rerun()
        
        # Theme selector
        theme_options = ["Light", "Dark", "Krishna", "Shiva"]
        selected_theme = st.selectbox("Choose Theme", theme_options, index=theme_options.index(st.session_state.theme) if st.session_state.theme in theme_options else 0)
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        
        # Language selector placeholder (would connect to translation API in full version)
        st.selectbox("Language", ["English", "Hindi", "Sanskrit", "Bengali", "Tamil", "Telugu"])
        
        # Display reading stats
        st.subheader("Your Reading Progress")
        stats = get_reading_stats()
        for scripture, stat in stats.items():
            st.progress(stat["percentage"] / 100)
            st.write(f"{scripture}: {stat['read']}/{stat['total']} ({stat['percentage']:.1f}%)")
    
    # Main content based on selected tab
    if st.session_state.current_tab == "Home":
        show_home_page()
    elif st.session_state.current_tab == "Scripture Explorer":
        show_scripture_explorer()
    elif st.session_state.current_tab == "Search & Discover":
        show_search_page()
    elif st.session_state.current_tab == "Reading Progress":
        show_reading_progress()
    elif st.session_state.current_tab == "Personal Notes":
        show_personal_notes()
    elif st.session_state.current_tab == "Meditation Timer":
        show_meditation_timer()

def show_home_page():
    st.title("🕉️ Divine Insights - Hindu Scriptures Explained")
    st.markdown("### Explore spiritual wisdom through verses, AI-generated summaries, and emotional insights.")
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📚 Scripture Explorer")
        st.markdown("Browse through sacred texts including Bhagavad Gita, Upanishads, and Yoga Sutras.")
        if st.button("Explore Scriptures"):
            st.session_state.current_tab = "Scripture Explorer"
            st.rerun()
    
    with col2:
        st.markdown("### 🔍 Search & Discover")
        st.markdown("Find verses by keywords, themes, or concepts across all scriptures.")
        if st.button("Search Verses"):
            st.session_state.current_tab = "Search & Discover"
            st.rerun()
    
    with col3:
        st.markdown("### ⏱️ Meditation Timer")
        st.markdown("Practice meditation with guidance from verses and a customizable timer.")
        if st.button("Start Meditating"):
            st.session_state.current_tab = "Meditation Timer"
            st.rerun()
            
    # Daily wisdom
    st.markdown("---")
    st.header("✨ Daily Wisdom")
    if st.button("Show Me a Random Verse"):
        scripture = random.choice(list(scriptures.keys()))
        verse_key = random.choice(list(scriptures[scripture].keys()))
        random_verse = scriptures[scripture][verse_key]

        st.markdown(f"**{scripture} - {verse_key}**")
        st.code(random_verse["sanskrit"], language="hi")
        st.write(random_verse["translation"])
        
        if "explanation" in random_verse:
            st.markdown("### Context & Meaning")
            st.write(random_verse["explanation"])
            
        # Display themes
        if "themes" in random_verse:
            st.markdown("### Themes")
            st.write(", ".join(random_verse["themes"]))
        
        # Actions for this verse
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Mark as Read"):
                mark_verse_as_read(scripture, verse_key)
        with col2:
            if st.button("Add to Favorites"):
                add_to_favorites(scripture, verse_key)
    
    # User testimonials
    st.markdown("---")
    st.header("💬 User Testimonials")
    
    testimonials = [
        {"name": "Arjun", "text": "This app has deepened my understanding of the Bhagavad Gita like never before."},
        {"name": "Priya", "text": "The AI summaries helped me grasp complex philosophical concepts in simple terms."},
        {"name": "Ravi", "text": "I use the meditation timer daily with verse reflections. It's transformed my practice."}
    ]
    
    columns = st.columns(3)
    for i, testimonial in enumerate(testimonials):
        with columns[i]:
            st.markdown(f"❝ {testimonial['text']} ❞")
            st.markdown(f"*— {testimonial['name']}*")

def show_scripture_explorer():
    st.title("Scripture Explorer")
    st.markdown("Browse and study verses from different Hindu scriptures with detailed explanations and insights.")
    
    # Scripture selection
    selected_scripture = st.selectbox("Choose a Scripture", list(scriptures.keys()))
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        difficulty_filter = st.multiselect("Filter by Difficulty", 
                                  ["Beginner", "Intermediate", "Advanced"], 
                                  default=["Beginner", "Intermediate", "Advanced"])
    
    with col2:
        # Get all unique themes across selected scripture
        # Get all unique themes across selected scripture
        all_themes = set()
        for verse_data in scriptures[selected_scripture].values():
            if "themes" in verse_data:
                all_themes.update(verse_data["themes"])
        
        theme_filter = st.multiselect("Filter by Theme", sorted(list(all_themes)), default=[])
    
    # Apply filters
    filtered_verses = {}
    for verse_key, verse_data in scriptures[selected_scripture].items():
        # Apply difficulty filter
        if "difficulty" in verse_data and verse_data["difficulty"] not in difficulty_filter:
            continue
        
        # Apply theme filter
        if theme_filter and not any(theme in verse_data.get("themes", []) for theme in theme_filter):
            continue
        
        filtered_verses[verse_key] = verse_data
    
    # Display filtered verses
    if not filtered_verses:
        st.warning("No verses match your filters. Try adjusting your selection.")
    else:
        st.subheader(f"Showing {len(filtered_verses)} verses from {selected_scripture}")
        
        for verse_key, verse_data in filtered_verses.items():
            # Check if this verse has been read
            is_read = is_verse_read(selected_scripture, verse_key)
            
            # Create expander with visual indicator for read status
            expander_label = f"{verse_key} {'✓' if is_read else ''}"
            with st.expander(expander_label):
                # Display verse content
                st.markdown("#### Sanskrit")
                st.code(verse_data["sanskrit"], language="sanskrit")
                
                st.markdown("#### Translation")
                st.write(verse_data["translation"])
                
                if "explanation" in verse_data:
                    st.markdown("#### Explanation")
                    st.write(verse_data["explanation"])
                
                # Display themes
                if "themes" in verse_data:
                    st.markdown("#### Themes")
                    st.write(", ".join(verse_data["themes"]))
                
                # Actions for this verse
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("Mark as Read", key=f"read_{selected_scripture}_{verse_key}"):
                        mark_verse_as_read(selected_scripture, verse_key)
                
                with col2:
                    if st.button("Add to Favorites", key=f"fav_{selected_scripture}_{verse_key}"):
                        add_to_favorites(selected_scripture, verse_key)
                
                with col3:
                    # Add comment section
                    if st.button("Add Comment", key=f"comment_btn_{selected_scripture}_{verse_key}"):
                        st.session_state[f"show_comment_{selected_scripture}_{verse_key}"] = True
                
                with col4:
                    # Export verse to image
                    if st.button("Export Image", key=f"export_{selected_scripture}_{verse_key}"):
                        img_buf = export_verse_to_image(selected_scripture, verse_key, verse_data)
                        st.download_button(
                            label="Download Image",
                            data=img_buf,
                            file_name=f"{selected_scripture}_{verse_key.replace(' ', '_')}.png",
                            mime="image/png",
                            key=f"download_{selected_scripture}_{verse_key}"
                        )
                
                # Display comment form if button was clicked
                if f"show_comment_{selected_scripture}_{verse_key}" in st.session_state and st.session_state[f"show_comment_{selected_scripture}_{verse_key}"]:
                    with st.form(key=f"comment_form_{selected_scripture}_{verse_key}"):
                        comment = st.text_area("Your Comment:", key=f"comment_{selected_scripture}_{verse_key}")
                        submit_button = st.form_submit_button("Submit Comment")
                        if submit_button and comment:
                            add_comment(selected_scripture, verse_key, comment)
                            st.session_state[f"show_comment_{selected_scripture}_{verse_key}"] = False
                            st.rerun()
                
                # Display existing comments
                comments = get_comments(selected_scripture, verse_key)
                if comments:
                    st.markdown("#### Comments")
                    for i, comment in enumerate(comments):
                        st.markdown(f"**{comment['username']}** - {comment['date']}")
                        st.write(comment['text'])
                        if i < len(comments) - 1:
                            st.markdown("---")
                
                # Find similar verses section
                st.markdown("#### Similar Verses")
                similar = find_similar_verses(selected_scripture, verse_key)
                if similar:
                    for i, verse in enumerate(similar[:3]):  # Show top 3
                        similarity_percentage = f"{verse['similarity'] * 100:.1f}%" if 'similarity' in verse else "Related"
                        st.markdown(f"**{verse['scripture']} - {verse['verse_key']}** ({similarity_percentage})")
                        st.write(verse['text'])
                        if i < len(similar[:3]) - 1:
                            st.markdown("---")
                else:
                    st.write("No similar verses found.")

def show_reading_progress():
    st.title("📚 Reading Progress")
    
    # Reading statistics
    stats = get_reading_stats()
    st.header("Your Reading Journey")
    
    # Overall progress
    total_verses = sum(stat["total"] for stat in stats.values())
    total_read = sum(stat["read"] for stat in stats.values())
    overall_percentage = (total_read / total_verses * 100) if total_verses > 0 else 0
    
    st.subheader(f"Overall Progress: {total_read}/{total_verses} verses ({overall_percentage:.1f}%)")
    st.progress(overall_percentage / 100)
    
    # Per scripture progress
    for scripture, stat in stats.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{scripture}**: {stat['read']}/{stat['total']} ({stat['percentage']:.1f}%)")
            st.progress(stat["percentage"] / 100)
        with col2:
            if stat["total"] > stat["read"]:
                if st.button(f"Read Next in {scripture}", key=f"next_{scripture}"):
                    # Find unread verse in this scripture
                    for verse_key in scriptures[scripture]:
                        if not is_verse_read(scripture, verse_key):
                            st.session_state.current_tab = "Scripture Explorer"
                            # This is simplistic - in a real app you'd want to pass parameters to show this specific verse
                            st.rerun()
                            break
    
    # Recently read verses
    if st.session_state.reading_progress:
        st.header("Recent Reading History")
        
        # Sort by date (most recent first)
        recent_reads = sorted(
            [(key.split("|")[0], key.split("|")[1], date) for key, date in st.session_state.reading_progress.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        for scripture, verse_key, date in recent_reads[:10]:  # Show last 10
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.write(f"**{scripture}**")
                st.write(f"*{verse_key}*")
            with col2:
                # Display shortened translation
                translation = scriptures[scripture][verse_key]["translation"]
                if len(translation) > 100:
                    translation = translation[:97] + "..."
                st.write(translation)
            with col3:
                st.write(f"Read on: {date.split()[0]}")
    else:
        st.info("You haven't marked any verses as read yet. Start your reading journey in the Scripture Explorer!")

def show_personal_notes():
    st.title("📝 Personal Notes & Favorites")
    
    tab1, tab2 = st.tabs(["My Favorites", "My Comments"])
    
    with tab1:
        if st.session_state.favorites:
            # Sort by date added (most recent first)
            sorted_favorites = sorted(
                st.session_state.favorites,
                key=lambda x: x["date_added"],
                reverse=True
            )
            
            for favorite in sorted_favorites:
                scripture = favorite["scripture"]
                verse_key = favorite["verse_key"]
                
                with st.expander(f"{scripture} - {verse_key}"):
                    verse_data = scriptures[scripture][verse_key]
                    
                    st.markdown("#### Translation")
                    st.write(verse_data["translation"])
                    
                    if "explanation" in verse_data:
                        st.markdown("#### Explanation")
                        st.write(verse_data["explanation"])
                    
                    st.write(f"*Added to favorites on: {favorite['date_added']}*")
                    
                    # Remove from favorites button
                    if st.button("Remove from Favorites", key=f"remove_{scripture}_{verse_key}"):
                        st.session_state.favorites = [f for f in st.session_state.favorites
                                                     if not (f["scripture"] == scripture and f["verse_key"] == verse_key)]
                        st.success(f"Removed '{verse_key}' from favorites!")
                        st.rerun()
        else:
            st.info("You haven't added any verses to your favorites yet.")
    
    with tab2:
        # Check if there are any comments
        has_comments = any(st.session_state.user_comments.values())
        
        if has_comments:
            # Flatten comments structure for display
            all_comments = []
            for key, comments in st.session_state.user_comments.items():
                scripture, verse_key = key.split("|")
                for comment in comments:
                    all_comments.append({
                        "scripture": scripture,
                        "verse_key": verse_key,
                        "text": comment["text"],
                        "date": comment["date"]
                    })
            
            # Sort by date (most recent first)
            sorted_comments = sorted(all_comments, key=lambda x: x["date"], reverse=True)
            
            for comment in sorted_comments:
                scripture = comment["scripture"]
                verse_key = comment["verse_key"]
                
                with st.expander(f"{scripture} - {verse_key} ({comment['date']})"):
                    # Display verse text for context
                    verse_text = scriptures[scripture][verse_key]["translation"]
                    st.write(f"**Verse:** {verse_text}")
                    
                    st.markdown("#### Your Comment")
                    st.write(comment["text"])
                    
                    # Edit comment functionality could be added here
        else:
            st.info("You haven't added any comments yet.")

def show_meditation_timer():
    st.title("⏱️ Meditation Timer")
    st.markdown("Set your meditation duration and focus on a verse for contemplation.")
    
    # Timer settings
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Meditation Duration (minutes)", 1, 60, 10)
    with col2:
        include_bell = st.checkbox("Include interval bells", value=True)
        if include_bell:
            bell_interval = st.slider("Bell interval (minutes)", 1, 15, 5)
    
    # Verse selection for meditation focus
    st.subheader("Select a Focus Verse")
    
    selection_method = st.radio("Select verse by:", ["Random", "Choose scripture", "From favorites"])
    
    meditation_verse = None
    meditation_scripture = None
    meditation_verse_key = None
    
    if selection_method == "Random":
        if st.button("Get Random Verse"):
            meditation_scripture = random.choice(list(scriptures.keys()))
            meditation_verse_key = random.choice(list(scriptures[meditation_scripture].keys()))
            meditation_verse = scriptures[meditation_scripture][meditation_verse_key]
    
    elif selection_method == "Choose scripture":
        meditation_scripture = st.selectbox("Scripture", list(scriptures.keys()))
        meditation_verse_key = st.selectbox("Verse", list(scriptures[meditation_scripture].keys()))
        meditation_verse = scriptures[meditation_scripture][meditation_verse_key]
    
    elif selection_method == "From favorites":
        if st.session_state.favorites:
            favorite_options = [f"{fav['scripture']} - {fav['verse_key']}" for fav in st.session_state.favorites]
            selected_favorite = st.selectbox("Select from your favorites", favorite_options)
            if selected_favorite:
                parts = selected_favorite.split(" - ")
                meditation_scripture = parts[0]
                meditation_verse_key = " - ".join(parts[1:])  # Handle verse keys that might contain hyphens
                meditation_verse = scriptures[meditation_scripture][meditation_verse_key]
        else:
            st.info("You don't have any favorites yet.")
    
    # Display selected verse for meditation
    if meditation_verse:
        st.markdown("### Your Meditation Focus")
        st.markdown(f"**{meditation_scripture} - {meditation_verse_key}**")
        st.write(meditation_verse["translation"])
    
    # Timer controls
    start_col, stop_col = st.columns(2)
    
    with start_col:
        if st.button("Start Meditation"):
            # Convert minutes to seconds
            timer_seconds = duration * 60
            
            # Store start time in session state
            st.session_state.meditation_start_time = time.time()
            st.session_state.meditation_duration = timer_seconds
            st.session_state.meditation_active = True
            
            # For bell intervals
            if include_bell:
                st.session_state.bell_interval = bell_interval * 60
            else:
                st.session_state.bell_interval = None
                
            # Record meditation in history
            if meditation_verse:
                meditation_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": duration,
                    "verse": f"{meditation_scripture} - {meditation_verse_key}" if meditation_verse else "None"
                }
                if "meditation_history" not in st.session_state:
                    st.session_state.meditation_history = []
                st.session_state.meditation_history.append(meditation_entry)
    
    with stop_col:
        if st.button("Stop Meditation"):
            if "meditation_active" in st.session_state:
                st.session_state.meditation_active = False
    
    # Timer display
    if "meditation_active" in st.session_state and st.session_state.meditation_active:
        elapsed = time.time() - st.session_state.meditation_start_time
        remaining = max(0, st.session_state.meditation_duration - elapsed)
        
        # Progress bar
        progress = 1 - (remaining / st.session_state.meditation_duration)
        st.progress(min(1.0, progress))
        
        # Time remaining display
        mins, secs = divmod(int(remaining), 60)
        time_display = f"{mins:02d}:{secs:02d}"
        st.markdown(f"<h1 style='text-align: center;'>{time_display}</h1>", unsafe_allow_html=True)
        
        # Check for interval bells
        if st.session_state.bell_interval:
            elapsed_intervals = int(elapsed / st.session_state.bell_interval)
            # This is a placeholder - in a real app you'd use JavaScript to play sounds
            if elapsed_intervals > 0 and elapsed % st.session_state.bell_interval < 1:
                st.markdown("🔔 Interval bell (sound would play here)")
        
        # Auto-stop timer when done
        if remaining <= 0:
            st.balloons()
            st.success("Meditation complete!")
            st.session_state.meditation_active = False
        
        # Force refresh every second (this is a hacky way to update a timer in Streamlit)
        time.sleep(1)
        st.rerun()
    
    # Meditation history
    if "meditation_history" in st.session_state and st.session_state.meditation_history:
        st.subheader("Meditation History")
        history_df = pd.DataFrame(st.session_state.meditation_history)
        st.dataframe(history_df)
        
        # Stats
        total_sessions = len(st.session_state.meditation_history)
        total_minutes = sum(session["duration"] for session in st.session_state.meditation_history)
        st.info(f"You've completed {total_sessions} meditation sessions totaling {total_minutes} minutes.")

# Run the app
if __name__ == "__main__":
    main()

