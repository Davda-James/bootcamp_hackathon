import streamlit as st
import dotenv
import logging
import os
import json
from model_2 import DocumentRetriever,generate_fact_from_chunks
# Load environment variables
dotenv.load_dotenv()
# Path to the JSON file
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the full path to the corpus.json file
# CORPUS_JSON_PATH = os.path.join(parent_dir, 'corpus.json')
CORPUS_JSON_PATH = os.environ.get("CORPUS_JSON_PATH")

# Streamlit Page Configuration
st.set_page_config(
    page_title="QueryCraft",
    page_icon="imgs/avatar_streamly.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/A-X-Z-Y-T-E/ai_ml_hackathon",
        "Report a bug": "https://github.com/A-X-Z-Y-T-E/ai_ml_hackathon",
        "About": """
            ## QueryCraft
            ### Author: team_name
        """
    }
)

# Configure logging
logging.basicConfig(level=logging.INFO)

retriever = DocumentRetriever(CORPUS_JSON_PATH)


def display_evidence(evidence_list):
    for evidence in evidence_list:
        st.markdown(f"#### {evidence['title']}")
        st.markdown(f"<p style='color: #FF5349;'><strong>Author:</strong> <span style='color: white;'>{evidence['author']}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #FF5349;'><strong>Source:</strong> <span style='color: white;'>{evidence['source']}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #FF5349;'><strong>Category:</strong> <span style='color: white;'>{evidence['category']}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #FF5349;'><strong>Published at:</strong> <span style='color: white;'>{evidence['published_at']}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #FF5349;'><strong>Fact:</strong> <span style='color: white;'>{evidence['fact']}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: blue;'><a href='{evidence['url']}' style='color: blue;'>Read more</a></p>", unsafe_allow_html=True)
        st.markdown("---")  
# Divider between each evidence

def main():
    # Insert custom CSS for glowing effect and layout
    st.markdown(
        """
        <style>
        .css-1d391kg {
            min-width: 300px !important;  /* Default width */
            max-width: 1000px !important;  /* Maximum draggable width */
        }

        /* Ensure sidebar has full height and allow scrolling */
        section[data-testid="stSidebar"] > div {
            display: flex;
            flex-direction: column;
            height: 100vh;  /* Full height of the viewport */
        }

        /* Container to hold responses and scroll above the input */
        section[data-testid="stSidebar"] > div > .css-1l02zno {
            flex-grow: 1;
            overflow-y: auto;  /* Scrollable content */
            display: flex;
            flex-direction: column-reverse; /* Reverse order so answer appears above input */
        }

        /* Pin the input box to the bottom */
        # section[data-testid="stSidebar"] > div > div:last-child {
        #     margin-top: auto;
        #     padding-bottom: 30px; 
        # }

        </style>
        """,
        unsafe_allow_html=True,
    )

    evidence_list = []
    
    with st.sidebar:
        # Input query from the user
        chat_input = st.chat_input("Ask me something:")
        # chat_input = st.sidebar.markdown(f"<p style='text-align: right;'>{answer}</p>", unsafe_allow_html=True)

        # Show answer above the search bar
        if chat_input:
            try:
                relevant_chunks=retriever.retrieve_relevant_chunks(chat_input)
                result = generate_fact_from_chunks(relevant_chunks, chat_input)
                answer=result['answer']
                query_type=result['question_type']
                evidence_list=result['evidence_list']

            except Exception as e:
                answer = f"An error occurred: {str(e)}"

            # Print the answer above the search bar
            st.sidebar.markdown(f"<p style='text-align: left; color: red;'><strong>Query:</strong> <span style='color: white;'>{chat_input}</span></p>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<p style='text-align: right; color: red;'><strong>Answer:</strong> <span style='color: white;'>{answer}</span></p>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<p style='text-align: right; color: red;'><strong>Query Type:</strong> <span style='color: white;'>{query_type}</span></p>", unsafe_allow_html=True)
    
    # Display the evidence list on the right side (main page)
    if evidence_list:
        display_evidence(evidence_list)

if __name__ == "__main__":
    main()
