# ALS Buddy

An empathetic AI chatbot built to provide information, emotional support, and conversational guidance on **Amyotrophic Lateral Sclerosis (ALS)**.

---

## Project Overview

**ALS Buddy** is designed to help users learn about ALS — its **symptoms**, **progression**, **treatments**, and **coping strategies** — through a natural and supportive chat experience.

It uses an intelligent combination of:
- **LangChain + LangGraph** for conversational logic and flow control  
- **Retrieval-Augmented Generation (RAG)** for context-based medical responses  
- **HuggingFaceHub model (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`)** for efficient, empathetic text generation  
- **FastAPI** backend to handle chatbot requests  
- **Streamlit** frontend for a clean chat-based interface  
- **Conversation memory** to preserve dialogue context  

---

## ✨ Features

- **Knowledge-Aware Conversations:** Retrieves relevant ALS information from pre-scraped medical sources using RAG.  
- **Contextual Memory:** Maintains conversation history so users can ask follow-up questions.  
- **Empathetic Responses:** Uses soft tone generation and emotional understanding for sensitive discussions.  
- **FastAPI Backend:** Handles queries and orchestrates the LangGraph conversation flow.  
- **Streamlit Frontend:** Provides an interactive, chat-style web interface.  
- **Chat History Options:** Clear or download past conversations easily.  
- **Deployable Anywhere:** Works locally or on Streamlit Cloud, Render, or Hugging Face Spaces.

---

## Architecture & File Structure

Your current project layout (kept intentionally simple and flat):
```
ALS-CHATBOT/
│
├── RAG/
│ ├── api_main.py ← FastAPI backend with /ask endpoint
│ ├── chatbot_with_memory.py ← Adds memory and empathy layer
│ ├── langgraph_chatbot.py ← LangGraph flow logic using RAG + LLM
│ ├── rag_chain.py ← Retrieval-Augmented Generation setup
│ ├── rag_setup.py ← Initial vectorstore builder
│ ├── Chroma-vectorstore-test.py ← ChromaDB testing utilities
│ └── chroma_db/ ← Vector database (persistent)
│
├── webscrapped-data/
│ ├── als_articles_expanded.json ← Cleaned ALS data used for retrieval
│ └── scrape_als_articles.py ← Web scraping script for ALS sources
│
├── chat_ui.py ← Streamlit chatbot interface
├── preprocess_data.py ← Preprocess scraped ALS data
├── .env ← API keys and environment configs
├── .gitignore ← Ignored files and folders
├── README.md ← This documentation file
└── venv/ ← Python virtual environment
```

---

## Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Language Model** | HuggingFaceHub (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) |
| **Frameworks** | LangChain, LangGraph |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Vector Store** | ChromaDB |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 |
| **Environment Management** | Python-dotenv |
| **Memory** | ConversationBufferMemory (LangChain) |

---

## Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Mizbain-Fathima/ALS_Buddy.git
cd ALS_Buddy
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
```
Activate it:
Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

###4️⃣ Set Up Environment Variables
Create a .env file in the root folder:
```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 5️⃣ Initialize RAG Vectorstore (run once)
```bash
python RAG/rag_setup.py
```

### 6️⃣ Start the FastAPI Backend
```bash
python RAG/api_main.py
```

Your backend will start at → http://127.0.0.1:8000

Test in your browser:
```cpp
http://127.0.0.1:8000/
```

### 7️⃣ Start the Streamlit Frontend

Open a new terminal and run:
```bash
streamlit run chat_ui.py
```

Visit → http://localhost:8501

### 💬 Example Queries
| User Input                              | Sample Response                                                                                                                        |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| *“What are the early symptoms of ALS?”* | Early symptoms include muscle weakness, stiffness, and difficulty speaking or swallowing. These symptoms gradually progress over time. |
| *“Is ALS curable?”*                     | Currently, there’s no complete cure, but several treatments can slow progression and improve quality of life.                          |
| *“I’m anxious about ALS.”*              | It’s completely normal to feel anxious. Remember, emotional support and therapy can help manage this journey better. 💙                |

---

### 📘 API Reference
POST /ask

Description: Sends a user query and returns a generated answer.

Request:
```json
{
  "question": "What are the symptoms of ALS?"
}
```

Response:
```json
{
  "answer": "ALS symptoms often begin with muscle cramps, stiffness, or weakness in the arms or legs."
}
```

---

### How It Works (Internally)

- **Web Scraping:** Gathers ALS-related content from trusted medical websites.
- **Data Processing:** Preprocessed and chunked into embeddings.
- **Chroma Vector Store:** Stores embeddings for semantic retrieval.
- **RAG Pipeline:** Combines retrieved context + LLM response.
- **LangGraph Flow:** Controls the conversational steps and empathy.
- **Streamlit UI:** Provides a clean, chat-based front-end.

---

### Future Enhancements

- Add multilingual support for broader accessibility.
- Integrate text-to-speech for voice-based interaction.
- Introduce advanced memory persistence (Redis or SQLite).
- Extend topics to other neurodegenerative diseases.