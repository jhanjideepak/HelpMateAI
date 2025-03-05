import os
import fitz  # PyMuPDF
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import pickle
from sklearn.metrics.pairwise import cosine_similarity


os.environ["OPENAI_API_KEY"] = ""

# Define cache paths
FAISS_INDEX_PATH = "faiss_index"
TEXT_CHUNKS_PATH = "text_chunks.pkl"

# Load PDF and Extract Text
def load_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


# Process multiple PDFs
def load_all_pdfs(pdf_folder):
    all_text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            text = load_pdf(os.path.join(pdf_folder, filename))
            all_text += text + "\n"
    return all_text


# Load and split text using FAISS
# def create_faiss_index(pdf_folder):
#     text = load_all_pdfs(pdf_folder)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=128)
#     text_chunks = text_splitter.split_text(text)
#
#     # Create embeddings and FAISS index
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#     vector_store = FAISS.from_texts(text_chunks, embeddings)
#     return vector_store, text_chunks

def create_faiss_index(pdf_folder):
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXT_CHUNKS_PATH):
        # Load FAISS index and text chunks from cache
        print("Loading FAISS index and text chunks from cache...")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            OpenAIEmbeddings(model="text-embedding-ada-002"),
            allow_dangerous_deserialization=True  # Explicitly allow deserialization
        )
        with open(TEXT_CHUNKS_PATH, "rb") as f:
            text_chunks = pickle.load(f)
    else:
        # Process PDFs and create FAISS index
        print("Processing PDFs and creating FAISS index...")
        text = load_all_pdfs(pdf_folder)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=128)
        text_chunks = text_splitter.split_text(text)

        # Create embeddings and FAISS index
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_texts(text_chunks, embeddings)

        # Save FAISS index and text chunks to cache
        vector_store.save_local(FAISS_INDEX_PATH)
        with open(TEXT_CHUNKS_PATH, "wb") as f:
            pickle.dump(text_chunks, f)

    return vector_store, text_chunks


# Re-ranking function using cosine similarity
def re_rank_results(query_vector, retrieved_indices, text_chunks, embeddings_model):
    reranked_texts = []
    reranked_scores = []

    for i in retrieved_indices[0]:
        if i < len(text_chunks):
            chunk_text = text_chunks[i]
            chunk_embedding = embeddings_model.embed_query(chunk_text)

            # Compute cosine similarity between query and passage embeddings
            similarity_score = cosine_similarity(
                np.array([query_vector]), np.array([chunk_embedding])
            )[0][0]

            reranked_texts.append((chunk_text, similarity_score))

    # Sort retrieved passages by similarity score in descending order
    reranked_texts.sort(key=lambda x: x[1], reverse=True)

    return reranked_texts


# Retrieve most relevant passage and generate response. Used k=3 to retrieve top 3 results
def retrieve_answer(query, vector_store, text_chunks, prompt_template):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    query_vector = embeddings.embed_query(query)
    D, I = vector_store.index.search(np.array([query_vector]), k=3)

    # relevant_text = text_chunks[I[0][0]]
    # relevant_text = "\n".join([text_chunks[i] for i in I[0] if i < len(text_chunks)])

    # Re-rank results
    reranked_texts = re_rank_results(query_vector, I, text_chunks, embeddings)

    # Select the highest ranked passage
    best_passage = reranked_texts[0][0] if reranked_texts else ""


    llm = ChatOpenAI(model="gpt-4")
    prompt = prompt_template.format(context=best_passage, question=query)
    # response = llm.invoke(f"Context: {relevant_text}\n\nQuestion: {query}\n\nAnswer:")
    response = llm.invoke(prompt)


    return best_passage, response


# Main part
pdf_folder = "./lang_rag/"  # Folder containing PDFs
vector_store, text_chunks = create_faiss_index(pdf_folder)
query = "When will the insurance be reinstated?"

custom_prompt = """You are an AI assistant. Use the following context to answer the question accurately. 

Context:
{context}

Question:
{question}

Answer:
"""
retrieved_passage, llm_response = retrieve_answer(query, vector_store, text_chunks, custom_prompt)

print("Relevant Passage:", retrieved_passage)
print("LLM Answer:", llm_response)
