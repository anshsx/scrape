
import os
import requests
import re
import random
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API key from environment variables
genai.configure(api_key="AIzaSyCIXu3XkADDdgETLiCTVsF6XNR0_c1ZJWM")

Set up the model configuration
generation_config = {
    "temperature": 2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=(
        "Summarize this content from a website and extract all useful information from this. Create a long and detailed brief summary. "
        "Combine all texts and write the whole summary in a single paragraph. Avoid phrases that imply you are an AI."
    ),
)

def scrape_urls(urls):
    # Split the URLs and remove whitespace
    url_list = [url.strip() for url in urls.split(',') if url.strip()]
    results = []

    for url in url_list:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise error for bad responses

            # Use BeautifulSoup to parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract paragraphs and headers for content
            paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
            headers_h1 = [h1.get_text().strip() for h1 in soup.find_all('h1')]
            headers_h2 = [h2.get_text().strip() for h2 in soup.find_all('h2')]

            # Combine extracted content
            combined_content = paragraphs + headers_h1 + headers_h2

            # Extract image URLs but filter out logos, buttons, and irrelevant images
            images = []
            for img in soup.find_all('img'):
                src = img.get('src', '')
                alt = img.get('alt', '')
                # Filter based on src and alt attributes to remove non-content images
                if not any(keyword in src.lower() for keyword in ['logo', 'button', 'favicon', 'icon']):
                    images.append(src)

            # Skip URLs with no content (empty content)
            if not combined_content:
                continue

            results.append({
                'content': [item for item in combined_content if item],  # Filter empty content
                'images': images  # Filtered content images
            })
        except requests.RequestException as e:
            # Log and append error in case of issues with the request
            results.append({'error': str(e)})

    return results

def summarize_combined_content(extracted_data):
    # Combine content from all URLs into one large string
    combined_content = " ".join(
        content_item for url_data in extracted_data 
        for content_item in url_data.get('content', [])
    )
    
    # Skip summarization if there's no content to summarize
    if not combined_content.strip():
        return "No content available to summarize."

    # Start a chat session with the model
    chat_session = model.start_chat(history=[])

    # Send the combined content to the model and get the response
    response = chat_session.send_message(combined_content)
    return response.text

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    if not data or 'urls' not in data:
        return jsonify({'error': 'Please provide URLs in the request body.'}), 400

    input_urls = data['urls']
    extracted_data = scrape_urls(input_urls)

    # Collect all images from the extracted data
    all_images = []
    for data in extracted_data:
        if 'images' in data:
            all_images.extend(data['images'])

    # Get up to 25 unique random image URLs, or all if fewer than 25
    unique_images = list(set(all_images))  # Remove duplicate URLs
    random_images = random.sample(unique_images, k=min(25, len(unique_images)))  # Select random images, up to 25

    # Summarize the combined content from all URLs
    overall_summary = summarize_combined_content(extracted_data)

    # Prepare the JSON response with summary and image URLs
    response_json = {
        "summary": overall_summary,
        "images": random_images  # Send the selected random image URLs
    }

    return jsonify(response_json)

# This ensures Flask runs only if executed directly, useful for local testing
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
