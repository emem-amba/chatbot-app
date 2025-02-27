import streamlit as st
import openai
import asyncio
import aiohttp
import os
import fitz  # PyMuPDF for PDF parsing
import faiss
import numpy as np
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# OpenAI API Keys (Store in .env for security)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Website URL
BASE_URL = "https://www.cbn.gov.ng"
PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

# Load OpenAI model
openai.api_key = OPENAI_API_KEY

# Streamlit UI with Animated Avatar
avatars = {
    "idle": "🤖",
    "scraping": "🔄",
    "chatting": "💬",
}

# Async Web Scraping
async def fetch(session, url):
    """ Fetch HTML content from a URL """
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def get_pdf_links(session, url):
    """ Extract PDF links from an HTML page """
    html = await fetch(session, url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    pdf_links = []

    for link in soup.find_all("a", href=True):
        if link["href"].endswith(".pdf"):
            full_link = BASE_URL + link["href"] if link["href"].startswith("/") else link["href"]
            pdf_links.append(full_link)

    return pdf_links

async def download_pdf(session, pdf_url):
    """ Download a PDF file asynchronously """
    pdf_name = pdf_url.split("/")[-1]
    pdf_path = os.path.join(PDF_FOLDER, pdf_name)

    if os.path.exists(pdf_path):  # Skip already downloaded PDFs
        return pdf_path

    try:
        async with session.get(pdf_url) as response:
            with open(pdf_path, "wb") as f:
                f.write(await response.read())
        return pdf_path
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """ Extract text from PDF using PyMuPDF """
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

async def scrape_cbn():
    """ Scrape the CBN website and extract text from PDFs """
    st.session_state.avatar = avatars["scraping"]

    async with aiohttp.ClientSession() as session:
        pdf_links = []

        # Define key sections of the website
        target_urls = [BASE_URL + "/out/", BASE_URL + "/IntOps/", BASE_URL + "/NewsArchive/", BASE_URL + "/FeaturedArticles/", BASE_URL + "/PaymentsSystem/", BASE_URL + "/Contacts/", BASE_URL + "/FOI/", BASE_URL + "/Documents/", BASE_URL + "/FAQS/", BASE_URL + "/Supervision/"]

        # Fetch PDF links concurrently
        tasks = [get_pdf_links(session, url) for url in target_urls]
        results = await asyncio.gather(*tasks)

        for result in results:
            pdf_links.extend(result)

        print(f"Found {len(pdf_links)} PDFs")

        # Download PDFs asynchronously
        download_tasks = [download_pdf(session, pdf) for pdf in pdf_links]
        pdf_paths = await asyncio.gather(*download_tasks)

    # Use multi-threading for PDF processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(extract_text_from_pdf, pdf_paths)

    return {pdf: text for pdf, text in zip(pdf_paths, results) if text}

def embed_text(texts):
    """ Convert text to embeddings using OpenAI """
    response = openai.embeddings.create(model="text-embedding-ada-002", input=texts)
    vectors = [data.embedding for data in response.data] 
    return np.array(vectors, dtype="float32")

def create_faiss_index(pdf_texts):
    """ Create a FAISS index for fast searching """
    text_list = list(pdf_texts.values())
    embeddings = embed_text(text_list)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, list(pdf_texts.keys())

def search_faiss(query, faiss_index, pdf_keys):
    """ Search the FAISS index for the most relevant document """
    query_embedding = embed_text([query])
    distances, indices = faiss_index.search(query_embedding, 1)

    return pdf_keys[indices[0][0]] if indices[0][0] < len(pdf_keys) else None

def chat_with_openai(query, pdf_text):
    """ Send query and relevant document text to GPT-4-Turbo """
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI expert on CBN regulations."},
            {"role": "user", "content": f"Using this document:\n{pdf_text}\nAnswer: {query}"}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("📊 CBN Chatbot")
st.markdown(f"## {st.session_state.get('avatar', avatars['idle'])}")

if st.button("Scrape CBN Website"):
    pdf_texts = asyncio.run(scrape_cbn())
    faiss_index, pdf_keys = create_faiss_index(pdf_texts)
    st.session_state.pdf_texts = pdf_texts
    st.session_state.faiss_index = faiss_index
    st.session_state.pdf_keys = pdf_keys
    st.session_state.avatar = avatars["idle"]
    st.success("Scraping completed!")

query = st.text_input("Ask a question:")
if query and "faiss_index" in st.session_state:
    st.session_state.avatar = avatars["chatting"]
    relevant_pdf = search_faiss(query, st.session_state.faiss_index, st.session_state.pdf_keys)
    response = chat_with_openai(query, st.session_state.pdf_texts[relevant_pdf])
    st.session_state.avatar = avatars["idle"]
    st.write(response)
