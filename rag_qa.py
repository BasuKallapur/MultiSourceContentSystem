import os
import logging
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# MultiFormatRAG Class for handling multiple document formats and enabling Q&A
class MultiFormatRAG:
    def __init__(self, openai_api_key: str):
        # Mapping file extensions to their respective document loaders
        self.loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader
        }

        # Splitting text into smaller chunks for better processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Using HuggingFace embeddings for document vectorization
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Setting up logging for debugging and tracking
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_documents(self, directory_path: str) -> List[Dict]:
        """
        Loads and processes documents from the specified directory.
        """
        documents = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            file_extension = os.path.splitext(file)[1].lower()

            if file_extension in self.loader_map:
                try:
                    # Using the appropriate loader based on file type
                    loader = self.loader_map[file_extension](file_path)
                    docs = loader.load()
                    self.logger.info(f"Successfully loaded {file}")
                    documents.extend(docs)
                except Exception as e:
                    self.logger.error(f"Error loading {file}: {str(e)}")
                    continue
        return documents

    def process_documents(self, documents: List[Dict]) -> FAISS:
        """
        Processes loaded documents into a FAISS vector store for efficient retrieval.
        """
        if not documents:
            self.logger.warning("No documents to process. Skipping FAISS indexing.")
            return None

        texts = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

    def create_qa_chain(self, vectorstore: FAISS, api_key: str) -> RetrievalQA:
        """
        Creates a Retrieval-based Q&A chain using a conversational AI model.
        """
        system_prompt = (
            "You are a conversational AI chatbot developed to replicate Basavaraj C Kkallapur, "
            "an accomplished professional with a proven track record in digital strategy, "
            "product innovation, and strategic investment management. "
        )

        # Defining a structured prompt for Q&A
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                f"{system_prompt}\n\n"
                "Please answer the following question strictly based on the provided context. "
                "If the answer is not found in the context, say 'I don't know'.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            )
        )

        # Initializing the LLM model for the Q&A system
        llm = ChatGroq(
            api_key=api_key,  # Passing API key for authentication
            model="llama3-70b-8192",
            temperature=0,  # Ensuring consistent and factual answers
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Creating the retrieval-based Q&A chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt_template}
        )

        return qa_chain

    def query(self, qa_chain: RetrievalQA, question: str) -> str:
        """
        Processes a user query and retrieves an answer from the Q&A chain.
        """
        try:
            response = qa_chain.invoke(question)
            return response['result']
        except Exception as e:
            self.logger.error(f"Error during query: {str(e)}")
            return f"Error processing query: {str(e)}"
