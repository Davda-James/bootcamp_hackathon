# Image Chatbot

This Flask application uses image captioning and keyword extraction to perform a Google search based on the content of an uploaded image. It leverages the BLIP model for image captioning, SpaCy for natural language processing, and SerpAPI for executing Google searches.

## Features

- Upload an image and receive a generated caption.
- Extract keywords from the caption.
- Perform a Google search based on the extracted keywords.
- Display the search results, including images and links.

## Prerequisites

- Python 3.7 or higher
- Flask
- Pillow
- Transformers
- SpaCy
- SerpAPI
- `torch` (PyTorch)
- `torchvision` (if required by the model)
- Additional dependencies specified in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Davda-James/bootcamp_hackathon
   cd ps1_google_lens_pro_max
   python app.py
