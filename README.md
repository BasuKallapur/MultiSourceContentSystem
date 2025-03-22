# Multi-Source Content Intelligence System

## Overview
The **Multi-Source Content Intelligence System** is a robust information retrieval system designed to process diverse content sources, including YouTube videos, web pages, and documents. It leverages **RAG (Retrieval-Augmented Generation) architecture**, **multi-modal processing**, and **content extraction techniques** to deliver precise content analysis and intelligent information synthesis.  
[Visit the application here](https://multisourcecontentsystem-hidevs.up.railway.app)


## Features
- **YouTube Video Summarization**: Extracts and summarizes transcripts from YouTube videos.
- **Document Processing**: Supports PDFs, DOCX, TXT, CSV, HTML, and Markdown files.
- **RAG-Based Q&A System**: Provides intelligent answers by retrieving relevant information from processed sources.
- **Vector Search Optimization**: Uses FAISS and HuggingFace embeddings for efficient information retrieval.

## Technologies Used
- **Python**
- **Streamlit** (for UI)
- **LangChain**
- **FAISS** (Vector Database)
- **YouTube Transcript API** (for extracting video transcripts)
- **OpenAI/GROQ API** (for LLM-powered summarization and Q&A)
- **LLaMA 3 (70B, 8192 context length)**

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/BasuKallapur/MultiSourceContentSystemHiDevs.git
   cd MultiSourceContentSystemHiDevs
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. **YouTube Summarization**:
   - Enter a YouTube URL and an API key.
   - Extract and display the transcript.
   - Generate a concise summary with key timestamps.

2. **Document Processing & RAG Q&A**:
   - Upload PDF, DOCX, TXT, CSV, HTML, or MD files.
   - Initialize the RAG system to enable intelligent search and Q&A.
   - Ask queries and receive relevant responses.

## Future Enhancements
- **Improve Search Ranking**: Enhance retrieval precision by refining vector search and ranking algorithms.
- **Multi-Modal Embeddings**: Integrate multi-modal embeddings to improve response coherence across different content types.
- **Optimized Processing Speed**: Improve efficiency for handling large document sets.
- **Entity Recognition**: Implement Named Entity Recognition (NER) to structure extracted information better.

## Contributing
Contributions are welcome! 🚀 If you'd like to improve the project, feel free to:
- Fork the repository and create a new branch.
- Implement your changes and ensure they work properly.
- Submit a pull request with a clear description of the update.

For major changes, please open an issue first to discuss the proposed modifications.


