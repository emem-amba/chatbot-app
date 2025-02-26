import os
import requests
import pdfplumber
import openai
import google.generativeai as genai
import streamlit as st
from bs4 import BeautifulSoup
#from dotenv import load_dotenv
from streamlit_lottie import st_lottie

# Load API keys
#load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def fetch_lottie_animation(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# Function to scrape website

def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    return "Error: Unable to fetch data"

# Function to extract text from PDFs
def extract_text_from_pdf(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open("temp.pdf", "wb") as pdf_file:
            pdf_file.write(response.content)
        text = ""
        with pdfplumber.open("temp.pdf") as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        os.remove("temp.pdf")
        return text
    return "Error: Unable to fetch PDF"

# Function to get AI response
def get_ai_response(prompt, model="openai"):
    if model == "openai":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    elif model == "gemini":
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    return "Invalid AI model selected."

# Streamlit UI
st.set_page_config(page_title="Web Scraper Chatbot", layout="wide")

st.title("🌐 AI Chatbot with Web Scraping and PDF Parsing")

# Load and display animated avatar
lottie_url = "https://assets4.lottiefiles.com/packages/lf20_jcikwtux.json"
animation = fetch_lottie_animation(lottie_url)
if animation:
    st_lottie(animation, height=200)

# User input
url = st.text_input("Enter website URL to scrape:")
use_ai_model = st.selectbox("Select AI Model:", ["openai", "gemini"])
user_question = st.text_area("Ask a question:")

if st.button("Get Answer"):
    if url.endswith(".pdf"):
        text = extract_text_from_pdf(url)
    else:
        text = scrape_website(url)
    
    full_prompt = f"Extract relevant details from this text and answer the user's query: {text}\nUser Question: {user_question}"
    response = get_ai_response(full_prompt, model=use_ai_model)
    st.write("### Response:")
    st.success(response)
