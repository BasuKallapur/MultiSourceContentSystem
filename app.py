import streamlit as st
from youtube_summarizer import extract_video_id, extract_transcript_details, call_groq_api, chunk_transcript
from rag_qa import MultiFormatRAG
import os
import shutil
import re
import time

def clear_temp_folder():
    if os.path.exists("temp_docs"):
        for file in os.listdir("temp_docs"):
            file_path = os.path.join("temp_docs", file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

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

def typewriter_effect(text, speed=0.005):
    """
    Simulates a typewriter effect by displaying text character by character.
    :param text: The text to display.
    :param speed: The delay between characters (in seconds).
    """
    placeholder = st.empty()
    for i in range(len(text) + 1):
        placeholder.markdown(f"<div style='white-space: pre-wrap;'>{text[:i]}</div>", unsafe_allow_html=True)
        time.sleep(speed)

def main():
    st.set_page_config(page_title="Multi-Source Content Intelligence System", layout="wide")
    st.title("Multi-Source Content Intelligence System")
    initialize_session_state()

    with st.sidebar:
        st.title("Configuration")
        groq_api_key = st.text_input("Enter GROQ API Key:", type="password")
        youtube_link = st.text_input("Enter YouTube video URL")
        uploaded_files = st.file_uploader("Upload Training Documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'csv', 'html', 'md'])

        if groq_api_key and youtube_link:
            if st.button("Initialize YouTube Summary"):
                with st.spinner("Initializing YouTube Summary..."):
                    try:
                        st.session_state.rag_system = None
                        st.session_state.qa_chain = None
                        st.session_state.chat_history = []

                        video_id = extract_video_id(youtube_link)
                        if video_id:
                            st.session_state.video_id = video_id
                            transcript_text = extract_transcript_details(youtube_link)
                            if transcript_text and transcript_text[0][0] == "Error":
                                if "blocked" in transcript_text[0][1].lower():
                                    st.error("YouTube is blocking requests from this IP. Please try again later or use a VPN.")
                                else:
                                    st.error(transcript_text[0][1])  # Display the error message
                                st.session_state.transcript_text = None
                            else:
                                st.session_state.transcript_text = transcript_text
                                st.session_state.summary_generated = False
                                st.success("YouTube Summary initialized successfully!")
                        else:
                            st.error("Invalid YouTube URL. Please check the link.")
                    except Exception as e:
                        st.error(f"Error initializing YouTube Summary: {str(e)}")

        if groq_api_key and uploaded_files:
            if st.button("Initialize RAG System"):
                with st.spinner("Initializing RAG System..."):
                    try:
                        st.session_state.transcript_text = None
                        st.session_state.video_id = None
                        st.session_state.summary_generated = False

                        clear_temp_folder()
                        os.makedirs("temp_docs", exist_ok=True)
                        for file in uploaded_files:
                            with open(os.path.join("temp_docs", file.name), "wb") as f:
                                f.write(file.getvalue())

                        st.session_state.rag_system = MultiFormatRAG(groq_api_key)
                        documents = st.session_state.rag_system.load_documents("temp_docs")
                        vectorstore = st.session_state.rag_system.process_documents(documents)
                        st.session_state.qa_chain = st.session_state.rag_system.create_qa_chain(vectorstore, groq_api_key)
                        st.success("RAG System initialized successfully!")
                    except Exception as e:
                        st.error(f"Error initializing RAG System: {str(e)}")

    if groq_api_key:
        if st.session_state.transcript_text:
            st.header("YouTube Video Summarizer")
            st.image(f"http://img.youtube.com/vi/{st.session_state.video_id}/0.jpg", width=250)
            with st.expander("View Transcript"):
                for ts, text in st.session_state.transcript_text:
                    st.markdown(f"**[{ts}]** {text}")

            if st.button("Get Summary"):
                st.session_state.summary_generated = True
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
                with st.expander("Generated Summary", expanded=True):
                    for summary in final_summary:
                        typewriter_effect(summary)  # Apply typewriter effect to each summary part

                st.download_button("Download Summary", "\n".join(final_summary), file_name="summary.txt")

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