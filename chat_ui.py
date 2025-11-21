import streamlit as st
import requests

# --- Backend API URL ---
API_URL = "http://127.0.0.1:8000/ask"

# --- Streamlit UI ---
st.set_page_config(page_title="ALS Support Chatbot", page_icon="💬", layout="centered")

st.title("💬 ALS Support Chatbot")
st.caption("Empathetic AI assistant to talk about ALS symptoms, treatments, and emotional support 💙")

# --- Chat Container ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
if prompt := st.chat_input("Ask about ALS symptoms, treatment, or emotional support..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Send request to FastAPI ---
    try:
        response = requests.post(API_URL, json={"question": prompt})
        response.raise_for_status()
        answer = response.json().get("answer", "Sorry, I couldn't process that.")
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})