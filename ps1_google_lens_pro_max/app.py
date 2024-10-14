import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
from serpapi import GoogleSearch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text)
    keywords = []
    for chunk in doc.noun_chunks:
        filtered_chunk = [token.text for token in chunk if token.pos_ != "DET"]
        keywords.append(" ".join(filtered_chunk))
    return " ".join(keywords)

app = Flask(__name__)

UPLOAD_FOLDER = "uploaded_images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

SERP_API_KEY = os.getenv("SERP_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def process_image():
    if 'image' not in request.files or 'query' not in request.form:
        return jsonify({"error": "Image file or query missing"}), 400

    image_file = request.files['image']
    query = request.form['query']

    if image_file.filename == '':
        return jsonify({"error": "No selected image"}), 400

    try:
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        raw_image = Image.open(image_path).convert('RGB')

        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        print("Generated Caption:", caption)

        keywords = extract_keywords(caption)
        print(keywords)
        print("Extracted Keywords:", keywords)

        keywords_with_query = f"{keywords}"
        print(keywords)

        params = {
            "engine": "google",
            "q": keywords,
            "api_key": SERP_API_KEY,
            "num": 10
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        search_results = []
        for result in results.get("organic_results", []):
            search_results.append({
                "title": result.get('title', 'No title'),
                "link": result.get('link', 'No link'),
                "redirect_link": result.get('redirect_link', 'No redirect link'),
                "thumbnail": result.get('thumbnail')
            })

        for result in results.get("images_results", []):
            search_results.append({
                "title": "Image result",
                "link": result.get('link'),
                "redirect_link": result.get('source'),
                "thumbnail": result.get('thumbnail')
            })

        return render_template('results.html', keywords=keywords_with_query, results=search_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
