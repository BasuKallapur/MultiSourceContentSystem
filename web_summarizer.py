import requests
import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

def validate_url(url):
    """
    Validates if the provided URL is properly formatted.
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))

def extract_webpage_content(url):
    """
    Extracts content from a webpage using WebBaseLoader and BeautifulSoup.
    Returns both the raw text and structured content.
    """
    try:
        # Validate URL
        if not validate_url(url):
            return None, "Invalid URL format. Please provide a valid URL starting with http:// or https://"
        
        # Use WebBaseLoader to load the webpage
        loader = WebBaseLoader(web_paths=[url])
        documents = loader.load()
        
        if not documents:
            return None, "No content could be extracted from the webpage."
        
        # Extract the main content
        main_content = documents[0].page_content
        
        # Use BeautifulSoup for additional cleaning and structure extraction
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title found"
        
        # Extract main content (try different selectors)
        main_selectors = [
            'main', 'article', '.content', '.main-content', 
            '#content', '#main', '.post-content', '.entry-content'
        ]
        
        main_element = None
        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                break
        
        if main_element:
            structured_content = main_element.get_text(separator='\n', strip=True)
        else:
            # Fallback to body content
            body = soup.find('body')
            structured_content = body.get_text(separator='\n', strip=True) if body else main_content
        
        # Clean up the content
        structured_content = re.sub(r'\n\s*\n', '\n\n', structured_content)  # Remove excessive newlines
        structured_content = re.sub(r'\s+', ' ', structured_content)  # Normalize whitespace
        
        return {
            'title': title_text,
            'url': url,
            'content': structured_content,
            'raw_content': main_content
        }, None
        
    except requests.exceptions.RequestException as e:
        return None, f"Error accessing the webpage: {str(e)}"
    except Exception as e:
        return None, f"Error processing the webpage: {str(e)}"

def chunk_webpage_content(content, chunk_size=2000, chunk_overlap=200):
    """
    Splits webpage content into manageable chunks for processing.
    """
    if not content or len(content) < chunk_size:
        return [content] if content else []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    return chunks

def call_groq_api_for_webpage(api_key, prompt, content):
    """
    Calls GROQ API to summarize webpage content.
    """
    import requests
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "compound-beta",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    for attempt in range(3):
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from model.").strip()
            elif response.status_code == 429:
                wait_time = min(60, 2 ** attempt * 10)
                print(f"Rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return f"Error: {response.status_code}, {response.text}"
        except requests.exceptions.Timeout:
            return "Error: Request timeout. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"
    
    return "Error: Too many retries due to rate limit."

def summarize_webpage(api_key, url):
    """
    Main function to summarize a webpage.
    """
    # Extract content from webpage
    webpage_data, error = extract_webpage_content(url)
    
    if error:
        return None, error
    
    # Chunk the content
    chunks = chunk_webpage_content(webpage_data['content'])
    
    if not chunks:
        return None, "No content could be extracted from the webpage."
    
    # Generate summary for each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            prompt = f"Summarize this part {i+1} of {len(chunks)} of the webpage content. Focus on key points and main ideas:"
        else:
            prompt = "Provide a comprehensive summary of this webpage content. Include key points, main ideas, and important details:"
        
        summary = call_groq_api_for_webpage(api_key, prompt, chunk)
        summaries.append(summary)
    
    # If we have multiple chunks, create a final summary
    if len(summaries) > 1:
        combined_summaries = "\n\n".join([f"Part {i+1}: {summary}" for i, summary in enumerate(summaries)])
        final_prompt = "Create a comprehensive final summary of this webpage by combining all the parts. Focus on the most important information:"
        final_summary = call_groq_api_for_webpage(api_key, final_prompt, combined_summaries)
    else:
        final_summary = summaries[0]
    
    return {
        'title': webpage_data['title'],
        'url': webpage_data['url'],
        'summary': final_summary,
        'chunks_processed': len(chunks),
        'content_length': len(webpage_data['content'])
    }, None 