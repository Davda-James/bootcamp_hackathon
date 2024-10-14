# QueryCraft

**QueryCraft** is an AI-powered chatbot application designed to answer and return relevent articles given by the user's query, by retrieving relevant information from a predefined dataset. Built using Streamlit, this project integrates various AI models to enhance user interaction and provide accurate responses.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
    
## Features
- User-friendly interface for querying information.
- Customizable environment for different datasets.
- Most relevant document retreival using BM25 and SBERT
- Most relevant fact generated using t5-flant-large LLM Model





## Installation

To run this project, you need to have Python 3.x installed. Follow these steps to set up the environment:

1. ### Clone the repository: ###

   ```bash
   git clone https://github.com/Davda-James/bootcamp_hackathon/
   ```

2. ### Change Directory to PS2 ###
    ```bash
     cd  ps2_Hog_RAGger
     ```

3. ### Install requirements after setting up virtual env ###
    ```bash
    python -m venv myenv

    ./myenv/Scripts/activate
    ```

    ```bash
    pip install -r requirements.txt
    ```
4. ### Running the streamlit app ###
    #### Set the path of CORPUS_JSON_PATH in .env ####
    ```bash
        CORPUS_JSON_PATH="corpus.json"
    ```
    #### Running the app now #### 
    ```bash
    streamlit run ./final_app.py
    ```

