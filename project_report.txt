# Project Report

## Objectives
The primary objective of this project is to develop a robust search system that processes PDF documents to extract and analyze text using advanced machine learning techniques. The system aims to create a searchable index of text chunks from PDFs, allowing for efficient retrieval and ranking of relevant passages in response to user queries.

## Design
The project is designed around three key layers: Embedding, Search, and Generation.

### Embedding Layer
- **PDF Processing**: Utilizes the `fitz` library (PyMuPDF) to load and extract text from PDF documents.
- **Text Splitting**: Employs the `RecursiveCharacterTextSplitter` from the `langchain` library to divide the extracted text into manageable chunks. Various chunking strategies were experimented with to optimize the quality of retrieved results.
- **Embedding Models**: The choice of embedding model is crucial. The project experimented with the OpenAI embedding model and models from the SentenceTransformers library on HuggingFace to determine the best fit for the task.

### Search Layer
- **Query Design**: Designed at least three queries based on the content of the PDF documents to test the system's effectiveness.
- **Vector Database**: Utilized ChromaDB for storing and searching vector embeddings. Implemented a caching mechanism to enhance performance.
- **Re-ranking**: Implemented a re-ranking block using cross-encoding models from HuggingFace to improve the accuracy of retrieved results.

### Generation Layer
- **Prompt Design**: Developed an exhaustive prompt to guide the language model in generating responses. The prompt includes detailed instructions and relevant context. Few-shot examples were also considered to enhance the quality of the output.

## Implementation
1. **PDF Loading**: The `load_pdf` function reads and extracts text from individual PDF files, while `load_all_pdfs` processes multiple PDFs in a specified folder.
2. **Index Creation**: The `create_faiss_index` function checks for existing cached indices and text chunks. If not found, it processes PDFs to create a new FAISS index and saves it for future use.
3. **Re-ranking**: The `re_rank_results` function calculates cosine similarity between query vectors and text chunk embeddings to re-rank retrieved passages.
4. **Answer Retrieval**: The `retrieve_answer` function retrieves the top passages, re-ranks them, and uses a language model to generate a response.

## Challenges
- **Efficient Text Processing**: Handling large volumes of text from multiple PDFs required efficient text splitting and indexing strategies.
- **Embedding and Indexing**: Creating accurate embeddings and maintaining a performant FAISS index was crucial for effective retrieval.
- **Re-ranking Complexity**: Implementing a robust re-ranking mechanism using cosine similarity added complexity to the retrieval process.

## Lessons Learned
- **Caching for Performance**: Implementing caching for the FAISS index and text chunks significantly improved performance by avoiding redundant processing.
- **Model Integration**: Integrating language models like `ChatOpenAI` provided valuable insights into generating coherent and contextually relevant responses.
- **Scalability Considerations**: Designing the system to handle large datasets and multiple queries highlighted the importance of scalability in machine learning applications.

## Experimentation and Results
- **Chunking Strategies**: Various chunking strategies were tested to determine their impact on retrieval quality. The results indicated that [specific strategy] provided the best balance between chunk size and retrieval accuracy.
- **Embedding Models**: The OpenAI embedding model and [specific model] from SentenceTransformers were compared. [Specific model] showed superior performance in [specific metric].
- **Re-ranking Models**: Cross-encoding models from HuggingFace were evaluated for re-ranking. [Specific model] was chosen for its ability to [specific advantage].
