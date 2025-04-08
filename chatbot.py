import streamlit as st
import google.generativeai as genai

# --- Configure Gemini API ---
genai.configure(api_key="AIzaSyBOhgRTiocBUrMkeppn-VbUgbHVXj90jhQ")  # Replace with your actual API key

# --- Load Gemini Model ---
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Streamlit App Configuration ---
# st.set_page_config(page_title="DIP Assistant", layout="centered", page_icon="🤖")

# --- Title and Description ---
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🤖 Learn about DIP</h1>
    
  
""", unsafe_allow_html=True)

# --- Initialize Session State for Conversation ---
if "history" not in st.session_state:
    st.session_state.history = [
        {
            "role": "user",
            "parts": [
                "You are DIP Assistant, a friendly and helpful AI built to teach and support users with Digital Image Processing (DIP) concepts. Speak clearly and professionally."
            ]
        }
    ]

# --- Custom Response Handler ---
def custom_response(msg):
    msg = msg.lower().strip()
    if msg in ["what is your name?", "what's your name?", "who are you?"]:
        return "I am **DIP Assistant**, your AI guide in Digital Image Processing! 🤖"
    elif "who created you" in msg or "who made you" in msg:
        return "I was created by **Manish Kumar** to help students learn Digital Image Processing. 🎓"
    elif "what can you do" in msg:
        return "I can help you understand topics like filtering, edge detection, image transforms, and more in **Digital Image Processing**."
    elif "i don't have a name" in msg or "you don't have a name" in msg:
        return "Actually, I do! I'm called **DIP Assistant**, nice to meet you! 👋"
    return None

# --- Chat Input ---
user_input = st.chat_input("Ask something related to DIP...")

if user_input:
    # Store user message
    st.session_state.history.append({"role": "user", "parts": [user_input]})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Check for custom responses
    reply = custom_response(user_input)

    # Generate response if not custom
    if not reply:
        try:
            response = model.generate_content(st.session_state.history)
            reply = response.text
        except Exception as e:
            reply = f"⚠️ Error: {str(e)}"

    # Show and save bot reply
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.history.append({"role": "model", "parts": [reply]})


st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px; color: gray;'>
        © 2025 | DIP Assistant | Developed by <b>Manish Kumar</b>
    </p>
""", unsafe_allow_html=True)
