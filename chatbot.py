import streamlit as st
import openai
import os
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# OpenAI API Keys (Store in .env for security)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load OpenAI model
openai.api_key = OPENAI_API_KEY

BASE_URL = "https://www.cbn.gov.ng"

async def fetch(session, url):
    """Fetches the content of a given URL asynchronously."""
    try:
        async with session.get(url, timeout=10) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def scrape_links(url, session, visited):
    """Recursively scrapes all internal links and collects their text content."""
    if url in visited or not url.startswith(BASE_URL):
        return ""
    
    visited.add(url)
    print(f"Scraping: {url}")
    html = await fetch(session, url)
    if not html:
        return ""
    
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    
    # Find all internal links
    links = {urljoin(BASE_URL, a['href']) for a in soup.find_all('a', href=True)}
    
    # Recursively scrape discovered links
    tasks = [scrape_links(link, session, visited) for link in links if link.startswith(BASE_URL)]
    results = await asyncio.gather(*tasks)
    
    return text + "\n" + "\n".join(results)

async def scrape_website():
    """Main function to scrape the entire CBN website."""
    visited = set()
    async with aiohttp.ClientSession() as session:
        return await scrape_links(BASE_URL, session, visited)

def ask_openai(question, context):
    """Sends user question along with scraped data to OpenAI for response."""
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert on the Central Bank of Nigeria (CBN) website."},
            {"role": "user", "content": f"Context: {context}\n\nUser question: {question}"}
        ]
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("CBN Website Chatbot")

if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = "Scraping in progress..."
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    scraped_content = loop.run_until_complete(scrape_website())
    st.session_state.scraped_data = scraped_content

question = st.text_input("Ask a question about CBN:")
if question:
    with st.spinner("Thinking..."):
        answer = ask_openai(question, st.session_state.scraped_data[:5000])  # Limiting context size
        st.write(answer)
