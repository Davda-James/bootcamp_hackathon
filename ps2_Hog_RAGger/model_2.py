from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class DocumentRetriever:
    def __init__(self, corpus_file_path):
        # Load and preprocess the corpus only once
        self.corpus, self.tokenized_corpus = self.load_and_preprocess_corpus(corpus_file_path)
        
        # Load models
        self.model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

    def load_and_preprocess_corpus(self, file_path):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        with open(file_path, 'r') as f:
            corpus = json.load(f)

        # Break each document into passages and preprocess: remove stop words, apply lemmatization
         # Break each document into passages and preprocess: remove stop words, apply lemmatization
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)  # Adjust sizes if needed

        passage_corpus = []
    
    # Iterate over each article in the corpus and split into passages
        for article in corpus:
            body = article['body']
            # Split the body of the article into smaller passages
            passages = text_splitter.split_text(article['title']+body)
            
            # Append each passage with the corresponding metadata
            for passage in passages:
                passage_corpus.append({
                    'title': article['title'],
                    'author': article['author'],
                    'url': article['url'],
                    'source': article['source'],
                    'category': article['category'],
                    'published_at': article['published_at'],
                    'body': passage
                })

        # Tokenize and preprocess passages (stop word removal, lemmatization)
        tokenized_corpus = [
            [lemmatizer.lemmatize(word) for word in word_tokenize(article['body'].lower()) if word not in stop_words]
            for article in passage_corpus
        ]
        return passage_corpus, tokenized_corpus

    def retrieve_relevant_chunks(self, query, top_n=8,most_relevant=4):
        # Initialize BM25
        bm25 = BM25Okapi(self.tokenized_corpus)

        # Tokenize the query
        tokenized_query = word_tokenize(query.lower())

        # Get scores for the query using BM25
        scores = bm25.get_scores(tokenized_query)

        # Get the top N relevant chunks (passages)
        top_indices = scores.argsort()[-top_n:][::-1]
        top_passages = [self.corpus[i] for i in top_indices]

        # Encode the query and the top retrieved chunks
        query_embedding = self.model_sbert.encode(query, convert_to_tensor=True)
        chunk_embeddings = self.model_sbert.encode([chunk['body'] for chunk in top_passages], convert_to_tensor=True)

        # Compute cosine similarities between the query and the top chunks
        cosine_similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]

        # Get the indices of the chunks sorted by similarity
        sorted_indices = cosine_similarities.argsort(descending=True)

        # Retrieve the top chunks based on SBERT refinement
        refined_top_chunks = [top_passages[i] for i in sorted_indices][:most_relevant]
        # Dictionary to track the highest-scoring chunk per title
        unique_title_chunks = {}

    # Iterate through the refined chunks and pick the top-scoring chunk for each unique title
        for chunk in refined_top_chunks:
            title = chunk['title']
            # If the title is not already in the dictionary, add the chunk
            if title not in unique_title_chunks:
                unique_title_chunks[title] = chunk

        # Return the chunks as a list
        return list(unique_title_chunks.values())

        # return refined_top_chunks

def determine_question_type(query,tokenizer_llm,model_llm):
        prompt = f"What type of question is '{query}'? Is it an inference_query, comparison_query, temporal_query, or null_query?"
        input_ids = tokenizer_llm(prompt, return_tensors="pt").input_ids.to('cuda')

        with torch.no_grad():
            output_ids = model_llm.generate(input_ids, max_length=10)
        
        question_type = tokenizer_llm.decode(output_ids[0], skip_special_tokens=True)
        return question_type
def generate_fact_from_chunks(relevant_chunks, query, llm_model='google/flan-t5-large'):
    # Load the selected LLM and tokenizer
    tokenizer_llm = T5Tokenizer.from_pretrained(llm_model,legacy=False)
    model_llm = T5ForConditionalGeneration.from_pretrained(llm_model).to('cuda')

    # Combine relevant chunks for the final prompt
    combined_relevant_chunks = " ".join([chunk['body'] for chunk in relevant_chunks])
    
    # Determine the type of question
    question_type = determine_question_type(query,tokenizer_llm,model_llm)
    
    # Construct prompt for LLM to generate the final answer based on relevant chunks
    prompt = f"Query: {query}\nRelevant Information: {combined_relevant_chunks}\n Question type:{question_type}\n\n  Answer the query based on the information and question type provided. Answer in one or two words."
    input_ids = tokenizer_llm(prompt, return_tensors="pt").input_ids.to('cuda')

    with torch.no_grad():
        output_ids = model_llm.generate(input_ids, max_length=20)

    final_answer = tokenizer_llm.decode(output_ids[0], skip_special_tokens=True)

    for chunk in relevant_chunks:
    # Modify the prompt to emphasize generating a fact around 50 words
        fact_prompt = (
            f"Generate a concise fact based on the information: {chunk['body']}, "
            f"relevant to {chunk['title']}, in approximately 50 words. Do not print the title again."
        )
        
        # Encode the prompt
        fact_input_ids = tokenizer_llm(fact_prompt, return_tensors="pt").input_ids.to('cuda')
        
        # Generate the output with a max length suitable for about 50 words (usually around 50-70 tokens)
        with torch.no_grad():
            fact_output_ids = model_llm.generate(
                fact_input_ids,
                max_length=100,  # Adjust max_length for approx. 50 words
                no_repeat_ngram_size=3,  # Avoid repeated phrases
            )
        
        # Decode and store the generated fact
        chunk["fact"] = tokenizer_llm.decode(fact_output_ids[0], skip_special_tokens=True)
        
        # Remove the body to reduce payload
        del chunk["body"]

    # Create the evidence list
        
    evidence_list = [chunk for chunk in relevant_chunks]


    # Construct the result in a dictionary format
    final_result = {
        "query": query,
        "answer": final_answer,
        "question_type": question_type,
        "evidence_list": evidence_list
    }

    return final_result

# Example usage
if __name__ == "__main__":
    # Path to the corpus.json file
    corpus_file = 'corpus.json'

    # Initialize DocumentRetriever
    retriever = DocumentRetriever(corpus_file)

    # Query to retrieve relevant chunks
    query="Considering the information from an article in The New York Times about the band Used To Be Young's latest tour and a review in Rolling Stone discussing the standout performance of a particular member during a recent concert, which member of Used To Be Young was highlighted for their exceptional solo during the tour's opening night and also plays the instrument that begins with the letter 'B'?"
    # Retrieve relevant chunks
    relevant_chunks = retriever.retrieve_relevant_chunks(query)
    final_result = generate_fact_from_chunks(relevant_chunks, query,llm_model='google/flan-t5-large')
    print(final_result)