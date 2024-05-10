import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import aiohttp
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import trafilatura

# Constants
MODEL_NAME = "dangvantuan/sentence-camembert-base"
MAX_URLS = 15
BATCH_SIZE = 4


@st.cache_resource
def load_model():
    """Load and return the tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model


async def fetch(url, semaphore):
    """Fetch and extract content from a URL."""
    async with semaphore:
        try:
            loop = asyncio.get_event_loop()
            downloaded = await loop.run_in_executor(None, trafilatura.fetch_url, url)
            if downloaded is None:
                st.warning(f"Warning: Failed to download content for URL: {url}")
                return None

            extracted_text = trafilatura.extract(downloaded, no_fallback=True, include_comments=False,
                                                 include_tables=False, favor_precision=True)
            if extracted_text is None:
                st.warning(f"Warning: Failed to extract text for URL: {url}")
                return None

            return extracted_text
        except Exception as e:
            st.error(f"Error: {e} for URL: {url}")
            return None


async def extract_texts(urls):
    """Extract texts from the provided URLs."""
    semaphore = asyncio.Semaphore(10)
    tasks = [fetch(url, semaphore) for url in urls]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = {}
    for url, resp in zip(urls, responses):
        if isinstance(resp, Exception):
            st.error(f"Failed to fetch content for {url}: {resp}")
        elif resp:
            results[url] = resp[:512 * 5]  # Limit the length of extracted text
    return results


def process_url(url, sentence_embeddings, tokenizer, model, url_contents):
    """Process a URL and return its semantic proximity."""
    if url not in url_contents:
        return {"URL": url, "Error": "Content not found"}

    text = url_contents[url]
    if text:
        inputs_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs_text = model(**inputs_text)
            embeddings_text = outputs_text[0].mean(dim=1)
        cos_sim = torch.nn.functional.cosine_similarity(sentence_embeddings, embeddings_text)
        return {"URL": url, "Semantic Proximity (%)": round(cos_sim.item() * 100, 2)}
    else:
        return {"URL": url, "Error": "Unable to extract text from URL"}


def process_batch(urls, sentence_embeddings, tokenizer, model, url_contents):
    """Process a batch of URLs."""
    results = []
    for url in urls:
        result = process_url(url, sentence_embeddings, tokenizer, model, url_contents)
        if result:
            results.append(result)
    return results


def dashboard():
    """Display the Streamlit dashboard."""
    tokenizer, model = load_model()
    st.title("Semantic Proximity for a Batch of Pages")

    st.markdown("""
        ### Instructions:
        - **Enter source URL:** The URL from which to fetch and analyze content.
        - **Enter target URLs:** List of target URLs to compare with the source URL.
        - **Submit:** Click to compute semantic proximity.
    """)

    form = st.form(key='my_form')
    sentence = form.text_input("Enter source URL", "", help="URL of the source page to compare against target URLs.")
    urls = form.text_area("Enter target URLs", "", help="Enter each target URL on a new line.")
    submit_button = form.form_submit_button(label='Submit')

    if submit_button and sentence and urls:
        valid_urls = [url.strip() for url in urls.splitlines() if url.strip()]
        if len(valid_urls) > MAX_URLS:
            valid_urls = valid_urls[:MAX_URLS]
            st.warning(f"Only the first {MAX_URLS} URLs will be processed.")

        with st.spinner('Processing...'):
            sentence_content = asyncio.run(extract_texts([sentence.strip()]))
            sentence = sentence_content.get(sentence.strip(), "")
            if not sentence:
                st.error("Failed to extract content for the source URL.")
                return

            inputs_sentence = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs_sentence = model(**inputs_sentence)
                embeddings_sentence = outputs_sentence[0].mean(dim=1)

            url_contents = asyncio.run(extract_texts(valid_urls))
            batched_urls = [valid_urls[i:i + BATCH_SIZE] for i in range(0, len(valid_urls), BATCH_SIZE)]
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_batch = {
                    executor.submit(
                        partial(process_batch, batch, embeddings_sentence, tokenizer, model, url_contents)): batch
                    for batch in batched_urls}

            st.session_state.results = []
            for future in as_completed(future_to_batch):
                result = future.result()
                if result:
                    st.session_state.results.extend(result)

            if 'df_results' not in st.session_state or not st.session_state.df_results.equals(
                    pd.DataFrame(st.session_state.results)):
                st.session_state.df_results = pd.DataFrame(st.session_state.results)

            st.session_state.df_results = st.session_state.df_results.sort_values(by='Semantic Proximity (%)',
                                                                                  ascending=False)
            st.dataframe(st.session_state.df_results, use_container_width=True)


if __name__ == "__main__":
    dashboard()
