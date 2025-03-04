import streamlit as st
from youtube_summarizer import extract_video_id, extract_transcript_details, call_groq_api, chunk_transcript
from rag_qa import MultiFormatRAG
import os
import shutil
import re

# Function to clear the temp_docs folder
def clear_temp_folder():
    if os.path.exists("temp_docs"):
        for file in os.listdir("temp_docs"):
            file_path = os.path.join("temp_docs", file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Initialize Session State
def initialize_session_state():
    if 'transcript_text' not in st.session_state:
        st.session_state.transcript_text = None
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'summary_generated' not in st.session_state:
        st.session_state.summary_generated = False

# Main Function
def main():
    st.set_page_config(page_title="Multi-Source Content Intelligence System", layout="wide")
    st.title("Multi-Source Content Intelligence System")

    # Initialize Session State
    initialize_session_state()

    # Sidebar Configuration
    with st.sidebar:
        st.title("Configuration")
        groq_api_key = st.text_input("Enter GROQ API Key:", type="password")
        youtube_link = st.text_input("Enter YouTube video URL")
        uploaded_files = st.file_uploader("Upload Training Documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'csv', 'html', 'md'])

        # Button to initialize YouTube Summary
        if groq_api_key and youtube_link:
            if st.button("Initialize YouTube Summary"):
                with st.spinner("Initializing YouTube Summary..."):
                    try:
                        # Reset RAG System state
                        st.session_state.rag_system = None
                        st.session_state.qa_chain = None
                        st.session_state.chat_history = []

                        # Initialize YouTube Summary
                        video_id = extract_video_id(youtube_link)
                        if video_id:
                            st.session_state.video_id = video_id
                            transcript_text = extract_transcript_details(youtube_link)
                            st.session_state.transcript_text = transcript_text
                            st.session_state.summary_generated = False  # Reset summary state
                            st.success("YouTube Summary initialized successfully!")
                        else:
                            st.error("Invalid YouTube URL. Please check the link.")
                    except Exception as e:
                        st.error(f"Error initializing YouTube Summary: {str(e)}")

        # Button to initialize RAG System
        if groq_api_key and uploaded_files:
            if st.button("Initialize RAG System"):
                with st.spinner("Initializing RAG System..."):
                    try:
                        # Reset YouTube Summary state
                        st.session_state.transcript_text = None
                        st.session_state.video_id = None
                        st.session_state.summary_generated = False  # Reset summary state

                        # Clear the temp_docs folder before saving new files
                        clear_temp_folder()

                        # Save uploaded files to temp_docs folder
                        os.makedirs("temp_docs", exist_ok=True)
                        for file in uploaded_files:
                            with open(os.path.join("temp_docs", file.name), "wb") as f:
                                f.write(file.getvalue())

                        # Initialize RAG system
                        st.session_state.rag_system = MultiFormatRAG(groq_api_key)
                        documents = st.session_state.rag_system.load_documents("temp_docs")
                        vectorstore = st.session_state.rag_system.process_documents(documents)
                        st.session_state.qa_chain = st.session_state.rag_system.create_qa_chain(vectorstore, groq_api_key)
                        st.success("RAG System initialized successfully!")
                    except Exception as e:
                        st.error(f"Error initializing RAG System: {str(e)}")

    # Main Area
    if groq_api_key:
        # YouTube Video Summarizer
        if st.session_state.transcript_text:
            st.header("YouTube Video Summarizer")
            st.image(f"http://img.youtube.com/vi/{st.session_state.video_id}/0.jpg", width=250)
            with st.expander("View Transcript"):
                for ts, text in st.session_state.transcript_text:
                    st.markdown(f"**[{ts}]** {text}")

            if st.button("Get Summary"):
                st.session_state.summary_generated = True  # Mark summary as generated
                st.markdown("### Summary")
                full_summary = []
                transcript_chunks = chunk_transcript(st.session_state.transcript_text, chunk_size=20)
                for chunk in transcript_chunks:
                    with st.spinner("Generating summary..."):
                        summary_part = call_groq_api(
                            groq_api_key,
                            "Summarize the video while retaining key timestamps and explanations.", 
                            chunk
                        )
                        full_summary.append(summary_part)

                final_summary = full_summary
                with st.expander("Generated Summary", expanded=True):  # Ensure the summary is always expanded
                    for summary in final_summary:
                        for line in summary.split("\n"):
                            match = re.match(r"\[(\d{2}:\d{2})\] (.*)", line)
                            if match:
                                ts, text = match.groups()
                                st.markdown(f"**[{ts}]** {text}")
                            else:
                                st.markdown(line)

                st.download_button("Download Summary", "\n".join(final_summary), file_name="summary.txt")

        # RAG-Based Q&A
        if st.session_state.qa_chain is not None:
            st.header("RAG-Based Q&A")
            user_input = st.chat_input("Ask a question:")
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_system.query(st.session_state.qa_chain, user_input)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

                st.rerun()

            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.write(f"You: {message['content']}")
                else:
                    st.write(f"AI: {message['content']}")

if __name__ == "__main__":
    main()