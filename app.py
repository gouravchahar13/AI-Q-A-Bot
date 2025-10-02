import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import streamlit as st

load_dotenv(".env")
HF_TOKEN = os.getenv("API_KEY")
if not HF_TOKEN:
    st.error("Set API_KEY in your .env file")
    st.stop()

client = InferenceClient(token=HF_TOKEN)
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

def ask(prompt: str) -> str:
    resp = client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return resp.choices[0].message["content"]

st.set_page_config(page_title="Hugging face Q&A Bot", page_icon="🤖", layout="wide")
st.title(" AI Q&A Bot 🤖")


if "history" not in st.session_state:
    st.session_state.history = []


if st.session_state.history:
    for chat in st.session_state.history:
        st.markdown(
            f"<div style='text-align:right; background-color:#DCF8C6; padding:8px; border-radius:10px; margin:5px 0; max-width:70%; margin-left:auto;'>{chat['user']}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='text-align:left; background-color:#E8E8E8; padding:8px; border-radius:10px; margin:5px 0; max-width:70%; margin-right:auto;'>{chat['bot']}</div>",
            unsafe_allow_html=True
        )

st.write("---") 


with st.form(key="input_form", clear_on_submit=True):
    prompt = st.text_input("Type your message here...")
    submit = st.form_submit_button("Send")
    
    if submit and prompt:
        with st.spinner("Thinking..."):
            answer = ask(prompt)
        st.session_state.history.append({"user": prompt, "bot": answer})
        st.rerun()  
