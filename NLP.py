import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from llama_cpp import Llama
import openai
import base64
from streamlit_option_menu import option_menu
import os
import google.generativeai as genai
from anthropic import Anthropic
import re
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import uuid

# Initialize session state for each chatbot
chatbots = ["Llama", "ChatGPT", "Gemini", "Claude", "Mistral", "Qwen"]

# Initialize session state variables for each chatbot
for bot in chatbots:
    if f"messages_{bot}" not in st.session_state:
        st.session_state[f"messages_{bot}"] = []
    if f"current_conversation_name_{bot}" not in st.session_state:
        st.session_state[f"current_conversation_name_{bot}"] = "Chat 1"
    # Store unique conversation ID for each bot
    if f"current_conversation_id_{bot}" not in st.session_state:
        st.session_state[f"current_conversation_id_{bot}"] = None

# Set default chatbot
if "selected_bot" not in st.session_state:
    st.session_state.selected_bot = "Llama"

# Link selected bot's messages and conversation name to session state
st.session_state.messages = st.session_state[f"messages_{st.session_state.selected_bot}"]
st.session_state.current_conversation_name = st.session_state[
    f"current_conversation_name_{st.session_state.selected_bot}"
]

# Functions for managing conversation history
def load_conversation_history(bot_name):
    # Load conversation history for a specific bot
    return st.session_state.get(f"{bot_name}_history", [])

def save_conversation_history(bot_name, history):
    # Save updated conversation history for a specific bot
    st.session_state[f"{bot_name}_history"] = history

def add_to_conversation_history(bot_name, conversation_id, conversation_name, messages):
    # Add or update a conversation in the bot's history
    history = load_conversation_history(bot_name)
    
    # Check if the conversation already exists by ID
    for conversation in history:
        if conversation["id"] == conversation_id:
            # Update messages and name if conversation is found
            conversation["messages"] = messages
            conversation["name"] = conversation_name
            save_conversation_history(bot_name, history)
            return
    
    # If conversation does not exist, append a new entry
    history.append({
        "id": conversation_id,
        "name": conversation_name,
        "messages": messages
    })
    save_conversation_history(bot_name, history)

def delete_conversation(bot_name, conversation_id):
    # Delete a conversation by ID
    history = load_conversation_history(bot_name)
    updated_history = [conv for conv in history if conv["id"] != conversation_id]
    save_conversation_history(bot_name, updated_history)

def clear_all_conversations(bot_name):
    # Clear all conversations for a specific bot
    save_conversation_history(bot_name, [])

def generate_conversation_name_from_input(user_input):
    # Generate a conversation name from the first few words of user input
    words = user_input.strip().split()
    short_title = " ".join(words[:5])
    return short_title if short_title else "New Conversation"

def update_conversation_name_in_history(bot_name, conversation_id, new_name):
    # Update conversation name in the bot's history
    history = load_conversation_history(bot_name)
    for conv in history:
        if conv["id"] == conversation_id:
            conv["name"] = new_name
            break
    save_conversation_history(bot_name, history)

# Configure Gemini and Claude API Keys
os.environ["API_KEY"] = ""
genai.configure(api_key=os.environ["API_KEY"])
ANTHROPIC_API_KEY = ""
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# CSS for styling
st.markdown(
    """
    <style>
    .bot-greeting {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        font-size: 22px;
        font-weight: bold;
        color: #6a0dad;
        margin-bottom: 20px;
    }
    .bot-greeting img {
        width: 50px;
        height: auto;
    }
    .pdf-uploader-header {
        color: #0066cc;
        font-weight: bold;
    }
    .chat-header {
        text-align: left;
        margin-top: 30px;
        font-size: 22px;
        font-weight: bold;
        color: #ff4b4b;
    }
    div.stButton > button {
        margin: 5px;
        width: 90%;
    }
    .menu-icon::before {
    content: '\\1F916'; /* Unicode for robot emoji */
    font-size: 18px;
    margin-right: 8px;
    }
    .menu-icon {
        display: inline-flex;
        align-items: center;
    }
    .menu-title {
        font-size: 20px;
        margin: 0;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to encode images to Base64
def get_base64_image(image_path):
    # Encode an image as Base64 for display
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        st.warning(f"Image not found: {image_path}")
        return ""

# Function to export chat history to a text file
def export_chat_history_to_text(messages):
    # Convert chat history into plain text format
    text_lines = []
    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        text_lines.append(f"{role}: {content}")
        text_lines.append("")  # Add an empty line for spacing
    return "\n".join(text_lines)

# Download NLTK stopwords if not already downloaded
nltk.download("wordnet")            # for WordNet lemmatizer
nltk.download("averaged_perceptron_tagger")  # for pos_tag
nltk.download("punkt")              # for word_tokenize
nltk.download("stopwords")          # Download stopwords
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Utility function for POS tagging
def get_wordnet_pos(treebank_tag):
    # Convert TreeBank POS tags to WordNet POS tags for lemmatization
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun if unknown

# Text cleaning function
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and symbols
    text = re.sub(r"[^\w\s]", "", text)

    # Tokenize text
    words = word_tokenize(text)

    # POS tagging
    tagged_words = pos_tag(words)  # e.g. [('produced', 'VBN'), ...]

    # Lemmatize tokens
    lemmatized_words = []
    for word, tag in tagged_words:
        # Skip stopwords
        if word not in stop_words:
            # Convert POS tag and lemmatize
            wn_tag = get_wordnet_pos(tag)
            lemmatized_word = lemmatizer.lemmatize(word, pos=wn_tag)
            lemmatized_words.append(lemmatized_word)

    # Join tokens back into a string
    text = " ".join(lemmatized_words)

    return text

# Placeholder for bot logos
llama_logo_base64 = get_base64_image("llama.jpg") or ""
chatgpt_logo_base64 = get_base64_image("openai.jpg") or ""
gemini_logo_base64 = get_base64_image("gemini.jpg") or ""
claude_logo_base64 = get_base64_image("claude.jpg") or ""
mistral_logo_base64 = get_base64_image("mistral.jpg") or ""
qwen_logo_base64 = get_base64_image("qwen.jpg") or ""

# Initialize session state
if "messages_llama" not in st.session_state:
    st.session_state.messages_llama = []
if "messages_chatgpt" not in st.session_state:
    st.session_state.messages_chatgpt = []
if "messages_gemini" not in st.session_state:
    st.session_state.messages_gemini = []
if "messages_claude" not in st.session_state:
    st.session_state.messages_claude = []
if "messages_mistral" not in st.session_state:
    st.session_state.messages_mistral = []
if "messages_qwen" not in st.session_state:
    st.session_state.messages_qwen = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "show_full_text" not in st.session_state:
    st.session_state.show_full_text = False
if "selected_bot" not in st.session_state:
    st.session_state.selected_bot = "Llama"
if "current_conversation_name" not in st.session_state:
    # Automatically set the first chat as the default
    st.session_state.current_conversation_name = "Chat 1"
if "messages" not in st.session_state:
    # Automatically link messages to the first chat
    st.session_state.messages = []


# PDF Extraction Functions
def extract_text_with_pdfplumber(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n"
        # Return raw text directly without calling clean_text here
        return full_text.strip()
    except Exception as e:
        st.error(f"Error extracting PDF text with PDFPlumber: {e}")
        return ""

def extract_text_with_pymupdf(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
            # Extract raw text from the PDF
            text = "".join([pdf[page].get_text() + "\n\n" for page in range(len(pdf))])
        # Return raw text (no preprocessing)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting PDF text with PyMuPDF: {e}")
        return ""

def extract_text_with_pypdf2(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        # Extract raw text from each page
        text = "".join(page.extract_text() or "" for page in reader.pages)
        # Return raw text (no preprocessing)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting PDF text with PyPDF2: {e}")
        return ""

# Load Llama Model
@st.cache_resource(ttl=10800)  # Cache for 3 hours
def load_llm_model():
    model_path = r"C:/Users/User/.ai-navigator/models/meta-llama/llama-2-7b-chat-hf/Llama-2-7B-Chat_Q4_K_M.gguf"
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        return Llama(
            model_path=model_path,
            n_ctx=4096,
            n_batch=64,
            n_threads=max(os.cpu_count() // 2, 1),
            n_gpu_layers=-1,  # Full GPU acceleration
            main_gpu=0,  # Primary GPU
            verbose=False,
            timeout=30  # Add timeout to prevent hanging
        )
    except Exception as e:
        st.error(f"Error loading Llama model: {e}")
        return None
    
# Function to load the Mistral model
@st.cache_resource(ttl=10800)  # Cache for 3 hours
def load_mistral_model():
    model_path = r"C:\Users\User\.ai-navigator\models\mistralai\Mistral-7B-Instruct-v0.2\Mistral-7B-Instruct-v0.2_Q4_K_M.gguf"
    try:
        model = Llama(
            model_path=model_path,  # Correctly use model_path
            n_ctx=4096,
            n_batch=64,
            n_threads=max(os.cpu_count() // 2, 1),
            n_gpu_layers=-1,  # Full GPU acceleration
            main_gpu=0,  # Primary GPU
            verbose=False,
            timeout=30  # Add timeout to prevent hanging
        )
        return model
    except Exception as e:
        st.error(f"Error loading Mistral model: {e}")
        return None

@st.cache_resource(ttl=10800)  # Cache for 3 hours
def load_qwen_model():
    model_path = r"C:\Users\User\.ai-navigator\models\Qwen\Qwen1.5-7B-Chat\Qwen1.5-7B-Chat_Q4_K_M.gguf"
    try:
        model = Llama(
            model_path=model_path,  # Correctly use model_path
            n_ctx=4096,  # Set context size
            n_batch=64,
            n_threads=max(os.cpu_count() // 2, 1),
            n_gpu_layers=-1,  # Full GPU acceleration
            main_gpu=0,  # Primary GPU
            verbose=False,
            timeout=30  # Add timeout to prevent hanging
        )
        return model
    except Exception as e:
        st.error(f"Error loading Qwen model: {e}")
        return None

def chatgpt_response(prompt, context=""):
    try:
        openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error interacting with ChatGPT: {e}")
        return f"An error occurred: {str(e)}"

def gemini_response(prompt, context=""):
    try:
        chat = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 10000,
            },
        ).start_chat(history=[])

        full_prompt = f"The following is the content of a document:\n\n{context}\n\nQ: {prompt}"
        response = chat.send_message(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error interacting with Gemini: {e}")
        return f"An error occurred: {str(e)}"

def claude_response(prompt, context=""):
    try:
        messages = []
        if context:
            messages.append({"role": "user", "content": context[:10000]})
        messages.append({"role": "user", "content": prompt})
        
        response = claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            messages=messages
        )
        return response.content[0].text.strip()
    except Exception as e:
        st.error(f"Error interacting with Claude: {e}")
        return f"An error occurred: {str(e)}"
    
# Main Streamlit Application
def main():
    st.title("📄 Academic Chat and Content Generator")
    
    # Configure bot logos and greetings dynamically
    bot_logo_base64 = (
        llama_logo_base64 if st.session_state.selected_bot == "Llama" else
        chatgpt_logo_base64 if st.session_state.selected_bot == "ChatGPT" else
        gemini_logo_base64 if st.session_state.selected_bot == "Gemini" else
        mistral_logo_base64 if st.session_state.selected_bot == "Mistral" else
        qwen_logo_base64 if st.session_state.selected_bot == "Qwen" else
        claude_logo_base64
    )

    bot_greeting_text = (
        "Llama is ready to assist you!" if st.session_state.selected_bot == "Llama"
        else "ChatGPT 3.5 is at your service!" if st.session_state.selected_bot == "ChatGPT"
        else "Gemini is ready to provide insightful answers for your document!" if st.session_state.selected_bot == "Gemini"
        else "Mistral is ready to serve!" if st.session_state.selected_bot == "Mistral"
        else "Alibaba Qwen is here!" if st.session_state.selected_bot == "Qwen"
        else "Claude is ready to analyze and assist with your queries!"
    )
    st.markdown(
        f"""
        <div class='bot-greeting'>
            <img src="data:image/png;base64,{bot_logo_base64}" alt="Bot Logo">
            <span>{bot_greeting_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    library_name = (
        "PDFPlumber" if st.session_state.selected_bot in ["Llama", "Claude"] else
        "PyMuPDF" if st.session_state.selected_bot in ["ChatGPT", "Mistral"] else
        "PyPDF2"
    )
    st.markdown(f"<span class='pdf-uploader-header'>🔒 {library_name} PDF Uploader</span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    # Extract raw and cleaned text
    if uploaded_file:
        if st.session_state.selected_bot in ["Llama", "Claude"]:
            raw_pdf_text = extract_text_with_pdfplumber(uploaded_file)
        elif st.session_state.selected_bot in ["ChatGPT", "Mistral"]:
            raw_pdf_text = extract_text_with_pymupdf(uploaded_file)
        else:
            raw_pdf_text = extract_text_with_pypdf2(uploaded_file)
    
        st.session_state.raw_pdf_text = raw_pdf_text
        st.session_state.pdf_text = clean_text(raw_pdf_text)
    
        if st.session_state.raw_pdf_text:
            st.success(f"✅ PDF Processed with {library_name}")
        else:
            st.error("❌ Failed to process the PDF.")
    else:
        st.session_state.raw_pdf_text = ""
        st.session_state.pdf_text = ""
        st.session_state.show_full_text = False
    
    st.markdown("<div class='chat-header'>Interactive Chat</div>", unsafe_allow_html=True)
    

    global llm
    llm = load_llm_model()
    global mistral_model
    mistral_model = load_mistral_model()
    global qwen_model
    qwen_model = load_qwen_model()
    
    with st.sidebar:
        bot_name = st.session_state.selected_bot
        st.markdown(f"### {bot_name} Conversation History")
    
        # Load conversation history for the selected bot
        conversation_history = load_conversation_history(bot_name)
        
        if conversation_history:
            for idx, conversation in enumerate(conversation_history):
                # Display each conversation by its 'name'
                # but internally store/operate by 'id'.
                conv_id = conversation["id"]
                conv_name = conversation["name"]
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Use a button that sets the current conversation if clicked
                    if st.button(conv_name, key=f"{bot_name}_select_{conv_id}"):
                        # On click, set the current conversation state
                        st.session_state[f"messages_{bot_name}"] = conversation["messages"]
                        st.session_state[f"current_conversation_name_{bot_name}"] = conv_name
                        st.session_state[f"current_conversation_id_{bot_name}"] = conv_id
                        
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"{bot_name}_delete_{conv_id}"):
                        delete_conversation(bot_name, conv_id)
                        # If deleted conversation was current, reset
                        if conv_id == st.session_state.get(f"current_conversation_id_{bot_name}"):
                            st.session_state[f"messages_{bot_name}"] = []
                            st.session_state[f"current_conversation_name_{bot_name}"] = None
                            st.session_state[f"current_conversation_id_{bot_name}"] = None
                        st.rerun()
        else:
            st.markdown("No conversations found.")
        
        # Option to start a new chat
        if st.button("New Chat"):
            bot_name = st.session_state.selected_bot
            conversation_history = load_conversation_history(bot_name)
        
            # Generate a unique conversation ID
            conversation_id = str(uuid.uuid4())
            new_chat_name = f"Chat {len(conversation_history) + 1}"
        
            # Reset messages for the new conversation
            st.session_state[f"messages_{bot_name}"] = []
            
            # Set the new current conversation name + ID
            st.session_state[f"current_conversation_name_{bot_name}"] = new_chat_name
            st.session_state[f"current_conversation_id_{bot_name}"] = conversation_id
        
            # Add the new conversation to history
            add_to_conversation_history(bot_name, conversation_id, new_chat_name, [])
        
            st.rerun()
            
        if st.button("Export Chat History"):
            bot_name = st.session_state.selected_bot
            messages = st.session_state.get(f"messages_{bot_name}", [])
            
            if messages:  # Ensure there are messages to export
                # Convert chat history to text
                chat_text = export_chat_history_to_text(messages)
                
                # Encode the text as Base64 for download link
                text_base64 = base64.b64encode(chat_text.encode("utf-8")).decode("utf-8")
                href = f'<a href="data:text/plain;base64,{text_base64}" download="{bot_name}_chat_history.txt">Download Chat History as Text</a>'
                
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No chat history to export.")
                
            # Add blank space using HTML or empty text
        st.markdown("<br>", unsafe_allow_html=True)  # Add a single line of blank space
            
        current_bot = st.session_state.get("selected_bot", "Llama")
        new_bot = option_menu(
            menu_title="Select Your Chatbot",  # Add the menu title
            options=["Llama", "ChatGPT", "Gemini", "Claude", "Mistral", "Qwen"],
            icons=["robot", "robot", "robot", "robot", "robot", "robot"],  # Add icons for each bot
            menu_icon="cast",  # Set the menu icon for the bar
            default_index=["Llama", "ChatGPT", "Gemini", "Claude", "Mistral", "Qwen"].index(st.session_state.selected_bot),
            orientation="vertical",  # Set the orientation to horizontal
            key="bot_selector_vertical",
        )
        
        # Check if the bot has changed
        if new_bot != current_bot:
            # Update selected bot in session_state
            st.session_state.selected_bot = new_bot
            st.session_state[f"current_conversation_id_{new_bot}"] = None  # Reset ID 
            st.session_state[f"messages_{new_bot}"] = []
            st.session_state[f"current_conversation_name_{new_bot}"] = "Chat 1"
    
            st.rerun()
        
        if st.session_state.raw_pdf_text:
            st.markdown("## PDF Text Preview")
            st.text_area("Text Preview", st.session_state.raw_pdf_text[:500], height=150)

            # Expand/Collapse button in the sidebar
            if st.button("Expand/ Collapse Text", key="expand_button_sidebar"):
                st.session_state.show_full_text = not st.session_state.show_full_text
        else:
            st.write("No PDF uploaded or no text extracted yet.")
        
    # Retrieve messages for the active bot
    messages = st.session_state.get(f"messages_{current_bot}", [])
    
    if st.session_state.show_full_text and st.session_state.raw_pdf_text:
        st.subheader("Full Extracted Text")
        show_preprocessed = st.checkbox("Show Preprocessed Text", value=False, key="preproc_checkbox_main")
        if show_preprocessed:
            st.text_area(
                "Full Extracted Text (Preprocessed)",
                st.session_state.pdf_text,
                height=300
            )
        else:
            st.text_area(
                "Full Extracted Text (Raw)",
                st.session_state.raw_pdf_text,
                height=300
            )
    
    # Referencing them locally to simplify
    def process_message(input_message):
        bot_name = st.session_state.selected_bot
        # Retrieve the current conversation ID or create if needed
        current_conv_id = st.session_state.get(f"current_conversation_id_{bot_name}")
        current_conv_name = st.session_state.get(f"current_conversation_name_{bot_name}")
        context = st.session_state.get("pdf_text", "No PDF content available.")
        cleaned_prompt = clean_text(input_message)
        
        # If there's no current conversation ID yet, create one
        if not current_conv_id:
            current_conv_id = str(uuid.uuid4())
            st.session_state[f"current_conversation_id_{bot_name}"] = current_conv_id
            add_to_conversation_history(bot_name, current_conv_id, current_conv_name, [])
        
        # Add user message to session state
        st.session_state[f"messages_{bot_name}"].append({"role": "user", "content": input_message})
        with st.chat_message("user"):
            st.markdown(input_message)
            
        # If this is the first user message for the conversation,
        # Rename the conversation from "Chat X" to a new short name
        if len(st.session_state[f"messages_{bot_name}"]) == 1:
            new_name = generate_conversation_name_from_input(input_message)
            st.session_state[f"current_conversation_name_{bot_name}"] = new_name
            update_conversation_name_in_history(bot_name, current_conv_id, new_name)
        
        # Streaming logic for Llama, Mistral, Qwen
        if bot_name in ["Llama", "Mistral", "Qwen"]:
            if bot_name == "Llama":
                model = llm
                full_prompt = f"""[INST] <<SYS>>
You are a helpful AI assistant...
{context[:10000] if context else 'No additional context'}

User Query: {cleaned_prompt}
<</SYS>>
[/INST]"""
                params = dict(
                    prompt=full_prompt,
                    max_tokens=4096,
                    temperature=0.2,
                    top_p=0.95,
                    repeat_penalty=1.1,
                    echo=False,
                    stream=True
                )
            elif bot_name == "Mistral":
                model = mistral_model
                full_prompt = f"Context: {context[:10000]}\n\nUser: {cleaned_prompt}\n\nAssistant:"
                params = dict(
                    prompt=full_prompt,
                    max_tokens=4096,
                    temperature=0.7,
                    top_p=0.9,
                    echo=False,
                    stream=True
                )
            else:  # Qwen
                model = qwen_model
                full_prompt = f"Context: {context[:10000]}\n\nUser: {cleaned_prompt}\n\nAssistant:"
                params = dict(
                    prompt=full_prompt,
                    max_tokens=4096,
                    temperature=0.7,
                    top_p=0.9,
                    echo=False,
                    stream=True
                )
    
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                streamed_response = ""
    
                for chunk in model(**params):
                    token = chunk["choices"][0]["text"]
                    streamed_response += token
                    message_placeholder.markdown(streamed_response)
    
                assistant_response = streamed_response.strip()
                st.session_state[f"messages_{bot_name}"].append({"role": "assistant", "content": assistant_response})
    
        else:
            # Non-streaming bots (Claude, ChatGPT, Gemini)
            if bot_name == "ChatGPT":
                assistant_response = chatgpt_response(cleaned_prompt, context)
            elif bot_name == "Gemini":
                assistant_response = gemini_response(cleaned_prompt, context)
            else:  # Claude
                assistant_response = claude_response(cleaned_prompt, context)
    
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state[f"messages_{bot_name}"].append({"role": "assistant", "content": assistant_response})
        
        # Finally, save updated conversation
        add_to_conversation_history(
            bot_name,
            current_conv_id,
            st.session_state[f"current_conversation_name_{bot_name}"],
            st.session_state[f"messages_{bot_name}"]
        )

    # Display existing messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about the PDF or any other topic"):
        process_message(prompt)

    # Additional content generation buttons
    col1, col2 = st.columns(2)
    button_action = None
    
    with col1:
        if st.button("Generate Notes"):
            button_action = "Generate Study Notes in Point Forms"
    
    with col2:
        if st.button("Generate Objective Questions"):
            button_action = "Generate Objective Questions"
    
    col3, _ = st.columns(2)
    with col3:
        st.markdown("<div class='generate-btn'>", unsafe_allow_html=True)
        if st.button("Generate Subjective Questions"):
            button_action = "Generate QA"
        st.markdown("</div>", unsafe_allow_html=True)
    
    with _:
        # Slider for specifying the number of questions
        st.session_state.question_count = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="question_slider",
        )
    
    # Handle button actions
    if button_action == "Generate QA":
        process_message(f"Generate {st.session_state.question_count} subjective questions and answers based on the PDF.")
    elif button_action == "Generate Objective Questions":
        process_message(
            f"Generate {st.session_state.question_count} multiple-choice questions and answers based on the PDF.")
    elif button_action:
        process_message(button_action)

if __name__ == "__main__":
    main()