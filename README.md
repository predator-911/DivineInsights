# 🕉️ Divine Insights - Hindu Scriptures Explained

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20.0%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/predator-911/DivineInsights)

**Divine Insights** is an interactive web application that helps users explore, understand, and reflect on the profound wisdom of Hindu scriptures. Powered by Streamlit and enhanced with AI capabilities, this application makes ancient spiritual teachings accessible to everyone.

![Divine Insights Demo](https://raw.githubusercontent.com/predator-911/DivineInsights/main/assets/demo.png)

## ✨ Features

### 📚 Scripture Explorer
- **Browse multiple scriptures**: Access the Bhagavad Gita, Upanishads, and Yoga Sutras
- **Original Sanskrit text**: Study verses in their original form
- **Modern translations**: Clear, accessible translations of complex concepts
- **In-depth explanations**: Understand the deeper meaning behind each verse
- **Thematic categorization**: Filter by themes like devotion, karma, meditation, etc.
- **Difficulty levels**: Sort content based on your knowledge level (beginner to advanced)

### 🔍 Search & Discovery
- **Keyword search**: Find verses containing specific words or concepts
- **Semantic similarity**: Discover related verses across different scriptures
- **Theme-based exploration**: Browse verses by spiritual themes

### 📝 Personal Learning Journey
- **Reading progress tracking**: Monitor which verses you've studied
- **Favorites collection**: Save verses that resonate with you
- **Personal notes**: Add your own reflections and insights
- **Progress visualization**: View statistics on your reading journey

### ⏱️ Meditation Practice
- **Verse-focused meditation**: Select verses for contemplative practice
- **Customizable timer**: Set your preferred meditation duration
- **Interval bells**: Optional timing signals during longer sessions
- **Meditation history**: Track your practice consistency

### 🎨 Customization
- **Multiple themes**: Choose from Light, Dark, Krishna, or Shiva visual themes
- **Export functionality**: Save verses as images to share or review offline
- **Language options**: Interface prepared for multilingual support

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/predator-911/DivineInsights.git
   cd DivineInsights
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app** in your browser at `http://localhost:8501`

### Optional: AI-Enhanced Features

For advanced features like semantic similarity, sentiment analysis, and zero-shot classification:

```bash
pip install transformers torch sentence-transformers
```

## 📋 Usage Guide

### Exploring Scriptures
1. Navigate to the "Scripture Explorer" tab
2. Select a scripture from the dropdown menu
3. Filter verses by difficulty level and/or themes
4. Click on any verse to expand and view details

### Personal Learning
1. Mark verses as "Read" to track your progress
2. Add important verses to "Favorites" for quick access
3. Add personal comments to record your insights
4. Visit the "Reading Progress" tab to view your statistics

### Meditation Practice
1. Go to the "Meditation Timer" tab
2. Set your desired meditation duration
3. Select a verse for contemplation
4. Start the timer and focus on your practice

## 🛠️ Technical Overview

### Architecture
- **Frontend**: Streamlit (Python-based web app framework)
- **Data Structure**: JSON-like dictionaries for scripture storage
- **AI Components**: 
  - Transformers for text summarization and analysis
  - SentenceTransformers for semantic similarity
  - WordCloud for visualization

### Customization Options
- **Data Sources**: Easily extend with additional scriptures by modifying the `scriptures` dictionary
- **Themes**: Add custom visual themes in the `set_theme()` function
- **Features**: Modular design for adding new functionality

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Areas
- Adding more scriptures and verses
- Implementing multilingual support
- Enhancing AI-based insights
- Improving visualization and analysis features

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 About the Author

**Lakshya Kumar**

- GitHub: [@predator-911](https://github.com/predator-911)
- Portfolio: [https://v0-aiwms.vercel.app/]
- LinkedIn: [https://www.linkedin.com/in/lakshya-kumar-7b16252b4/]

## 🙏 Acknowledgements

- Sanskrit texts and translations sourced from academic and spiritual resources
- Special thanks to the Streamlit team for their excellent framework
- Gratitude to the open-source community for AI and NLP tools

---

<p align="center">
  Made with ❤️ and spiritual curiosity
</p>
