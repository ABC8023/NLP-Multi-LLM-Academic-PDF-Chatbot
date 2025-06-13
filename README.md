# 📄 Multi-LLM PDF Chatbot

A powerful academic assistant built with Streamlit that allows users to upload and interact with PDF documents using multiple large language models. It can extract text, clean and preprocess it using NLP techniques, and generate study notes, objective questions, and subjective Q&A. 

## 🚀 Features

- 📤 Upload and extract text from PDFs using PDFPlumber, PyMuPDF, or PyPDF2
- 🧠 Chat with multiple LLMs: LLaMA, ChatGPT, Claude, Gemini, Mistral, Qwen
- 🧹 Automatic text cleaning using NLTK
- 📝 Generate notes, MCQs, and subjective questions from PDF content
- 💬 Maintain per-bot conversation history
- 🔄 Switch between models seamlessly
- ⬇️ Export chat history as text
- 🖼️ Base64 image handling for bot logos
- 📚 Context-aware conversations based on uploaded PDF

---

## ⚙️ How It Works

1. **Upload PDF**: Users can upload any PDF document.
2. **Text Extraction**: The app extracts text using different libraries depending on the selected bot.
3. **NLP Processing**: Text is cleaned and lemmatized with NLTK to improve model responses.
4. **Chat Interface**: Users can chat with any selected bot. The bot uses the processed PDF content as context.
5. **Conversation History**: Each bot maintains its own chat history, which can be exported.
6. **Content Generation**: Buttons allow generating notes, MCQs, or Q&A based on the document.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
git clone https://github.com/yourusername/multi-llm-pdf-chatbot.git

cd multi-llm-pdf-chatbot 

# Create a Virtual Environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install Dependencies

pip install -r requirements.txt

# Add API Keys (for ChatGPT, Claude, Gemini)

OPENAI_API_KEY = "your-openai-key"
ANTHROPIC_API_KEY = "your-claude-key"
GEMINI_API_KEY = "your-gemini-key"

# Add Local Model Files

Place .gguf model files in the paths specified in the code, or update the paths in the script:

LLaMA: ~/.ai-navigator/models/meta-llama/...

Mistral: ~/.ai-navigator/models/mistralai/...

Qwen: ~/.ai-navigator/models/Qwen/...

# 🧪 Run the App

streamlit run NLP.py

# 📦 Dependencies

streamlit

pdfplumber, PyMuPDF, PyPDF2

nltk, re, uuid, base64

openai, anthropic, google.generativeai

llama-cpp-python

# 🧑‍💻 License

MIT License – feel free to fork, modify, and improve.
