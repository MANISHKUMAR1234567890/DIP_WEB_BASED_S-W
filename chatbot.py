import streamlit as st
import ollama

# Streamlit UI
# st.set_page_config(page_title="DIP Assistant", page_icon="🤖")
st.title("Learn about Image Procssing")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input field
user_input = st.chat_input("Type your message...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response from LLaMA
    response = ollama.chat(model="myllama", messages=[{"role": "user", "content": user_input}])

    # Display assistant message
    bot_reply = response["message"]["content"]
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
