import os
import faiss
import numpy as np
import pickle
import time
import json
import re
from posthog import api_key
import spacy
from tqdm import tqdm
from mistralai import Mistral
from typing import List, Dict, Any, Tuple, Union
from neo4j import GraphDatabase
import hashlib
import textwrap
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, Comment
import sqlite3, threading
from typing import Optional
# Type alias for clarity
Chunk = Dict[str, Any]

# Constants
mistral_api_key = 'VrAMhIHO61FjHTAYeibtLmla52bWnorV'
DEFAULT_CHUNK_SIZE = 500  # Target number of words per chunk
MAX_TOKENS_BEFORE_SECTION_SPLIT = 5000  # Max tokens before splitting by section
URL_REGEX = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
MD_URL_REGEX = r'<<(https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+)>>'  # Format: <<url>>
SIMPLIFIED_URL = r'<<.*?>>'
# Initialize spaCy model - download with: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Please download it with: python -m spacy download en_core_web_sm")
    nlp = None


def tag_visible(element):
    # Filter out unwanted elements
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def extract_visible_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')

        # Remove scripts, styles, etc. completely
        for tag in soup(['script', 'style', 'head', 'meta', 'noscript']):
            tag.decompose()

        # Get all text and filter
        texts = soup.find_all(string=True)
        visible_texts = filter(tag_visible, texts)

        # Join and clean text
        return '\n'.join(t.strip() for t in visible_texts if t.strip())

    except Exception as e:
        return f"Error: {e}"

class Summarizer:
    def __init__(self, api):
        self.model = Mistral(api_key=api)
        self.system_prompt = '''
Summarise the given context of some web page into a description of 3 to 4 line paragraph.
'''

    def summarise_text(self, text):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]
        for _ in range(3):  # retry up to 3 times
            try:
                response = self.model.chat.complete(
                    model='mistral-small-latest',
                    messages=messages
                )
                time.sleep(1)
                return response.choices[0].message.content
            except Exception as e:
                print(f"Retrying due to error: {e}")
                time.sleep(1)
        return "Failed to generate summary after retries."

summarizer = Summarizer(api = mistral_api_key)

SIMPLIFIED_URL = r"<<[^<>]+>>"
DB_PATH = "url_cache.db"
os.remove(DB_PATH) if os.path.exists(DB_PATH) else None  # Remove old DB if exists

def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS url_info (
            url TEXT PRIMARY KEY,
            fetched_at INTEGER,
            visible_text TEXT,
            summary TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_url_info(url: str, visible_text: str, summary: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO url_info (url, fetched_at, visible_text, summary)
        VALUES (?, ?, ?, ?)
    """, (url, int(time.time()), visible_text, summary))
    conn.commit()
    conn.close()


def get_url_info(url: str, db_path: str = DB_PATH) -> Optional[dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.execute("""
        SELECT fetched_at, visible_text, summary
          FROM url_info
         WHERE url = ?
    """, (url,))
    row = cur.fetchone()
    conn.close()
    if row:
        fetched_at, visible_text, summary = row
        return {
            "url": url,
            "fetched_at": fetched_at,
            "visible_text": visible_text,
            "summary": summary
        }
    return None


def extract_info_url(text: str) -> str:
    init_db()
    placeholders = set(re.findall(SIMPLIFIED_URL, text))
    for placeholder in placeholders:
        url = placeholder[2:-2]
        info = get_url_info(url)
        if info:
            visible_text = info["visible_text"]
            summary = info["summary"]
        else:
            visible_text = extract_visible_text(url)
            summary = summarizer.summarise_text(visible_text)
            save_url_info(url, visible_text, summary)
        replacement = f"{summary} {placeholder}"
        text = text.replace(placeholder, replacement)
    return text


class Neo4jConnector:
    """Class to manage Neo4j connections and operations."""
    
    def __init__(self, uri="neo4j+s://d4e98294.databases.neo4j.io", user="neo4j", password="CMB2JFluGdYmo5kNG2x7qeAA8krSJK32GTgAJogmYdA"):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j server URI
            user: Neo4j username
            password: Neo4j password
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print(f"Connected to Neo4j at {uri}")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def create_entity(self, entity_type, name, properties=None):
        """
        Create an entity node in Neo4j.
        
        Args:
            entity_type: Type of entity (Person, Organization, etc.)
            name: Name of the entity
            properties: Additional properties
        """
        if not self.driver:
            return None
            
        with self.driver.session() as session:
            props = properties or {}
            # Create unique ID based on name and type
            entity_id = hashlib.md5(f"{name}:{entity_type}".encode()).hexdigest()
            
            query = (
                f"MERGE (e:{entity_type} {{id: $id, name: $name}}) "
                "SET e += $properties "
                "RETURN e"
            )
            
            result = session.run(
                query, 
                id=entity_id,
                name=name, 
                properties=props
            )
            return result.single()
    
    def create_relationship(self, source_type, source_name, target_type, target_name, rel_type, properties=None):
        """
        Create a relationship between two entities.
        
        Args:
            source_type: Type of source entity
            source_name: Name of source entity
            target_type: Type of target entity
            target_name: Name of target entity
            rel_type: Type of relationship
            properties: Additional properties
        """
        if not self.driver:
            return None
            
        with self.driver.session() as session:
            props = properties or {}
            
            # Create relationship
            query = (
                f"MATCH (a:{source_type}), (b:{target_type}) "
                "WHERE a.name = $source_name AND b.name = $target_name "
                "MERGE (a)-[r:" + rel_type + "]->(b) "
                "SET r += $properties "
                "RETURN r"
            )
            
            result = session.run(
                query, 
                source_name=source_name, 
                target_name=target_name,
                properties=props
            )
            return result.single()
    
    def store_chunk(self, chunk_id, metadata):
        """
        Store a document chunk in Neo4j.
        
        Args:
            chunk_id: Unique ID for the chunk
            metadata: Chunk metadata
        """
        if not self.driver:
            return None
            
        with self.driver.session() as session:
            # Store the chunk as a Document node
            query = (
                "MERGE (d:Document {id: $id}) "
                "SET d += $metadata "
                "RETURN d"
            )
            
            result = session.run(
                query, 
                id=chunk_id,
                metadata={k: v for k, v in metadata.items() if k != 'content'}  # Don't store full content
            )
            return result.single()
    
    def link_entity_to_chunk(self, entity_type, entity_name, chunk_id):
        """
        Link an entity to a document chunk.
        
        Args:
            entity_type: Type of entity
            entity_name: Name of entity
            chunk_id: ID of the chunk
        """
        if not self.driver:
            return None
            
        with self.driver.session() as session:
            query = (
                f"MATCH (e:{entity_type}), (d:Document) "
                "WHERE e.name = $entity_name AND d.id = $chunk_id "
                "MERGE (e)-[r:MENTIONED_IN]->(d) "
                "RETURN r"
            )
            
            result = session.run(
                query, 
                entity_name=entity_name, 
                chunk_id=chunk_id
            )
            return result.single()

class FaissEmbedder:
    """
    A class for embedding and storing unstructured data using FAISS.
    Replaces ChromaDB for the ingestion part.
    """
    
    def __init__(self, 
                api_key: str, 
                base_path: str = "faiss_dbs", 
                embedding_model: str = "mistral-embed",
                neo4j_uri: str = "neo4j+s://d4e98294.databases.neo4j.io",
                neo4j_user: str = "neo4j",
                neo4j_password: str = "CMB2JFluGdYmo5kNG2x7qeAA8krSJK32GTgAJogmYdA"):
        """
        Initialize the FAISS embedder.
        
        Args:
            api_key: Mistral API key
            base_path: Base directory to store FAISS indices
            embedding_model: Name of the embedding model to use
            neo4j_uri: Neo4j server URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.api_key = api_key
        self.base_path = base_path
        self.embedding_model = embedding_model
        self.mistral_client = Mistral(api_key=api_key)
        
        # Create base directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Track currently loaded indices
        self.indices = {}
        self.mappings = {}
        
        # Initialize Neo4j connector
        self.neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
        
        # Create URL store directory
        self.url_store_path = os.path.join(base_path, "url_store")
        os.makedirs(self.url_store_path, exist_ok=True)
        
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'neo4j') and self.neo4j:
            self.neo4j.close()
        
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts using Mistral API.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to embed in each batch
            
        Returns:
            numpy array of embeddings
        """
        embeddings = []
        
        # Smaller batch size to avoid rate limits
        batch_size = min(batch_size, 16)
        
        # Add delay between batches to avoid rate limits
        batch_delay = 1.0  # Start with 1 second delay between batches
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i + batch_size]
            
            # Add retry logic for API failures
            max_retries = 5  # Increased max retries
            for attempt in range(max_retries):
                try:
                    # Add delay before making API call
                    time.sleep(batch_delay)
                    
                    # Use the correct parameter name 'inputs' as used in retriver.py
                    response = self.mistral_client.embeddings.create(
                        model=self.embedding_model,
                        inputs=batch
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    # If successful, we can slightly decrease the delay (but keep a minimum)
                    batch_delay = max(1.0, batch_delay * 0.9)
                    break
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"Embedding attempt {attempt+1} failed: {error_message}")
                    
                    # If rate limited, increase delay with exponential backoff
                    if "429" in error_message or "rate limit" in error_message.lower():
                        # Exponential backoff: 2^attempt seconds
                        retry_delay = min(60, 2 ** (attempt + 2))  # Max 60 second delay
                        batch_delay = min(60, batch_delay * 2)  # Double the delay between batches
                        print(f"Rate limit hit. Waiting {retry_delay}s and increasing batch delay to {batch_delay}s")
                        time.sleep(retry_delay)
                    else:
                        # For other errors, wait a shorter time
                        time.sleep(2 * (attempt + 1))
                    
                    if attempt == max_retries - 1:
                        raise
        
        return np.array(embeddings, dtype="float32")
    
    def create_index(self, group_id: int, texts: List[str], metadata: List[Dict] = None) -> None:
        """
        Create a FAISS index for a group of texts.
        
        Args:
            group_id: Identifier for the group
            texts: List of text chunks to index
            metadata: Optional metadata for each text chunk
        """
        if not texts:
            print(f"No texts provided for group {group_id}")
            return
        
        # Create directory for this group if it doesn't exist
        group_dir = os.path.join(self.base_path, f"group_{group_id}")
        os.makedirs(group_dir, exist_ok=True)
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        
        # Choose index type based on corpus size
        if len(texts) > 10000:
            # For large corpora, use HNSW index which is more efficient for search
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
        else:
            # For smaller corpora, use flat index which is exact but slower for large datasets
            index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Save index
        index_path = os.path.join(group_dir, "index.faiss")
        faiss.write_index(index, index_path)
        
        # Create and save mapping (text -> id)
        if metadata is None:
            metadata = [{"id": i, "text": text} for i, text in enumerate(texts)]
        else:
            # Ensure each metadata entry has 'id' and 'text' fields
            for i, (meta, text) in enumerate(zip(metadata, texts)):
                meta["id"] = i
                meta["text"] = text
        
        mapping_path = os.path.join(group_dir, "mapping.pkl")
        with open(mapping_path, "wb") as f:
            pickle.dump(metadata, f)
        
        # Also save text content separately for easy access
        texts_path = os.path.join(group_dir, "texts.pkl")
        with open(texts_path, "wb") as f:
            pickle.dump(texts, f)
        
        print(f"Created FAISS index for group {group_id} with {len(texts)} texts")
        
        # Store in memory for immediate use
        self.indices[group_id] = index
        self.mappings[group_id] = metadata
    
    def load_index(self, group_id: int) -> Tuple[Any, List[Dict]]:
        """
        Load a FAISS index for a group.
        
        Args:
            group_id: Identifier for the group
            
        Returns:
            Tuple of (FAISS index, mapping dictionary)
        """
        if group_id in self.indices and group_id in self.mappings:
            return self.indices[group_id], self.mappings[group_id]
        
        group_dir = os.path.join(self.base_path, f"group_{group_id}")
        
        if not os.path.exists(group_dir):
            raise FileNotFoundError(f"No index found for group {group_id}")
        
        # Load index
        index_path = os.path.join(group_dir, "index.faiss")
        index = faiss.read_index(index_path)
        
        # Load mapping
        mapping_path = os.path.join(group_dir, "mapping.pkl")
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)
        
        # Store in memory for future use
        self.indices[group_id] = index
        self.mappings[group_id] = mapping
        
        return index, mapping
    
    def search(self, group_id: int, query: str, k: int = 5) -> List[Dict]:
        """
        Search a group for similar texts.
        
        Args:
            group_id: Identifier for the group
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        # Load index and mapping if not already loaded
        try:
            index, mapping = self.load_index(group_id)
        except FileNotFoundError:
            print(f"No index found for group {group_id}")
            return []
        
        # Embed query
        query_embedding = self.embed_texts([query])[0].reshape(1, -1)
        
        # Search
        distances, indices = index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(mapping):
                result = mapping[idx].copy()
                result["distance"] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def delete_index(self, group_id: int) -> bool:
        """
        Delete a FAISS index for a group.
        
        Args:
            group_id: Identifier for the group
            
        Returns:
            True if successful, False otherwise
        """
        group_dir = os.path.join(self.base_path, f"group_{group_id}")
        
        if not os.path.exists(group_dir):
            print(f"No index found for group {group_id}")
            return False
        
        # Remove from memory
        if group_id in self.indices:
            del self.indices[group_id]
        
        if group_id in self.mappings:
            del self.mappings[group_id]
        
        # Delete directory
        try:
            for file in os.listdir(group_dir):
                os.remove(os.path.join(group_dir, file))
            os.rmdir(group_dir)
            return True
        except Exception as e:
            print(f"Error deleting index for group {group_id}: {str(e)}")
            return False
    
    def get_all_groups(self) -> List[int]:
        """
        Get a list of all group IDs.
        
        Returns:
            List of group IDs
        """
        groups = []
        for item in os.listdir(self.base_path):
            if item.startswith("group_") and os.path.isdir(os.path.join(self.base_path, item)):
                group_id = int(item.replace("group_", ""))
                groups.append(group_id)
        return groups

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and names
        """
        if not nlp:
            print("SpaCy model not loaded. Skipping entity extraction.")
            return {}
            
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract named entities
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            # Avoid duplicates
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text.
        
        Args:
            text: Text to extract URLs from
            
        Returns:
            List of URLs
        """
        # First check for URLs in markdown format <<url>>
        md_urls = re.findall(MD_URL_REGEX, text)
        
        # Then check for regular URLs
        regular_urls = re.findall(URL_REGEX, text)
        
        # Filter out URLs that are already in markdown format to avoid duplicates
        filtered_regular_urls = [url for url in regular_urls if f"<<{url}>>" not in text]
        
        # Combine both lists, removing duplicates
        all_urls = list(set(md_urls + filtered_regular_urls))
        return all_urls
    
    def store_url(self, url: str, context: Dict[str, Any]) -> None:
        """
        Store a URL for later crawling.
        
        Args:
            url: URL to store
            context: Context of where the URL was found
        """
        # Create a unique filename for the URL
        domain = urlparse(url).netloc
        url_hash = hashlib.md5(url.encode()).hexdigest()
        filename = f"{domain}_{url_hash}.json"
        
        # Store URL data
        url_data = {
            "url": url,
            "discovered_at": time.time(),
            "crawled": False,
            "context": context
        }
        
        # Save to file
        with open(os.path.join(self.url_store_path, filename), "w") as f:
            json.dump(url_data, f)
            
    def format_urls_in_text(self, text: str) -> str:
        """
        Format URLs in text to use the <<url>> format.
        
        Args:
            text: Text to format
            
        Returns:
            Text with URLs in <<url>> format
        """
        # First check for URLs already in markdown format to avoid double formatting
        md_formatted = set(re.findall(MD_URL_REGEX, text))
        
        # Format regular URLs if they're not already formatted
        def replace_url(match):
            url = match.group(0)
            if f"<<{url}>>" in md_formatted:
                return url  # Don't reformat if already in markdown format
            return f"<<{url}>>"
        
        # Replace URLs with <<url>> format
        return re.sub(URL_REGEX, replace_url, text)

def chunk_and_index(corpus_item: Dict) -> List[Chunk]:
    """
    Chunk a document into sections and paragraphs intelligently.
    
    Args:
        corpus_item: Dictionary containing at least 'content' and 'filename' keys
    
    Returns:
        List of chunk dictionaries with metadata
    """
    content = corpus_item.get('content', '')
    filename = corpus_item.get('filename', 'unknown')
    
    if not content:
        return []
    
    # Format URLs in content with <<url>> format
    content = format_urls_in_content(content)
    
    chunks = []
    
    # Count approx tokens (rough estimate: 1 token = 4 chars)
    token_count = len(content) // 4
    
    # Step 1: If content is large, split by section headings
    if token_count > MAX_TOKENS_BEFORE_SECTION_SPLIT:
        # Split by markdown headings (##)
        sections = re.split(r'(^|\n)##\s+(.+?)$', content, flags=re.MULTILINE)
        
        # Process the split results
        sections_processed = []
        i = 0
        while i < len(sections):
            if i + 2 < len(sections) and (sections[i] == '' or sections[i] == '\n'):
                section_title = sections[i+1]
                section_content = sections[i+2]
                sections_processed.append((section_title, section_content))
                i += 3
            else:
                # Handle case where there's content before the first heading
                if i == 0 and sections[i] and sections[i] != '\n':
                    sections_processed.append(("Introduction", sections[i]))
                i += 1
    else:
        # For smaller documents, treat the whole thing as one section
        sections_processed = [("Main Content", content)]
    
    # Step 2: Process each section into paragraph chunks
    for section_idx, (section_title, section_content) in enumerate(sections_processed):
        # Add a chunk for the whole section
        section_chunk = {
            'id': f"{filename}_section_{section_idx}",
            'filename': filename,
            'level': 'section',
            'title': section_title,
            'content': section_content,
            'position': section_idx
        }
        chunks.append(section_chunk)
        
        # Now split into paragraphs
        # Split the section content into paragraphs by finding one or more blank lines
        # This regex pattern matches any newline followed by optional whitespace and another newline
        # Split section content into lines
        # lines = section_content.split('\n')
        
        # Group lines into paragraphs based on min 5 lines, max 30 lines with 1 line overlap
        # paragraphs = []
        # current_para = []
        
        # for line in lines:
        #     current_para.append(line)
            
            # # Check if we've reached the max paragraph size
            # if len(current_para) >= 30:
            #     paragraphs.append('\n'.join(current_para))
            #     # Keep the last line for overlap with the next paragraph
            #     current_para = [current_para[-1]] if current_para else []
            
            # # If we have at least 5 lines and encounter an empty line, consider it a paragraph break
            # elif len(current_para) >= 5 and line.strip() == '':
            #     paragraphs.append('\n'.join(current_para))
            #     # Keep the last line for overlap with the next paragraph
            #     current_para = [current_para[-1]] if current_para else []
        
        # Add the last paragraph if it exists
        # For paragraphs at the end of a section, we don't enforce the minimum line count
        # if current_para:
        #     paragraphs.append('\n'.join(current_para))
        paragraphs = re.split(r'\n\s*\n+', section_content)
        # Process each paragraph
        for para_idx, para in enumerate(paragraphs):
            if not para.strip():
                continue
                
            # Further break down long paragraphs
            words = para.split()
            para_chunks = []
            
            for i in range(0, len(words), DEFAULT_CHUNK_SIZE):
                para_chunk = ' '.join(words[i:i+DEFAULT_CHUNK_SIZE])
                para_chunks.append(para_chunk)
            
            # Create a chunk for each paragraph segment
            for chunk_idx, para_chunk in enumerate(para_chunks):
                para_chunk_dict = {
                    'id': f"{filename}_section_{section_idx}_para_{para_idx}_chunk_{chunk_idx}",
                    'filename': filename,
                    'level': 'paragraph',
                    'title': f"{section_title} - Paragraph {para_idx+1}.{chunk_idx+1}",
                    'content': para_chunk,
                    'section_title': section_title,
                    'section_idx': section_idx,
                    'para_idx': para_idx,
                    'chunk_idx': chunk_idx,
                    'position': para_idx * 1000 + chunk_idx  # For ordering
                }
                chunks.append(para_chunk_dict)
    
    return chunks

def format_urls_in_content(content: str) -> str:
    """
    Format all URLs in content to use the <<url>> format.
    
    Args:
        content: Text content to format
        
    Returns:
        Content with URLs in <<url>> format
    """
    # First find all URLs already in markdown format
    md_urls = set([f"<<{url}>>" for url in re.findall(MD_URL_REGEX, content)])
    
    # Find and format regular URLs that aren't already in markdown format
    def replace_url(match):
        url = match.group(0)
        formatted_url = f"<<{url}>>"
        if formatted_url in md_urls:
            return url  # Don't reformat if already in markdown format
        return formatted_url
    
    return re.sub(URL_REGEX, replace_url, content)

def index_chunks(chunks: List[Chunk], 
                api_key: str = None, 
                base_path: str = "faiss_dbs",
                neo4j_uri: str = "neo4j+s://d4e98294.databases.neo4j.io",
                neo4j_user: str = "neo4j",
                neo4j_password: str = "CMB2JFluGdYmo5kNG2x7qeAA8krSJK32GTgAJogmYdA") -> None:
    """
    Index a list of chunks using FAISS, spaCy for entity extraction, and Neo4j.
    
    Args:
        chunks: List of chunk dictionaries
        api_key: Mistral API key
        base_path: Base directory for FAISS indices
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
    """
    if not chunks:
        print("No chunks to index")
        return
    
    # Initialize embedder
    embedder = FaissEmbedder(
        api_key=api_key, 
        base_path=base_path,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )
    
    # Ensure all chunks have URLs formatted correctly
    for chunk in chunks:
        if 'content' in chunk:
            chunk['content'] = extract_info_url(chunk['content'])
            chunk['content'] = format_urls_in_content(chunk['content'])
    
    # Group chunks by level
    section_chunks = [c for c in chunks if c['level'] == 'section']
    paragraph_chunks = [c for c in chunks if c['level'] == 'paragraph']
    
    # Process section chunks
    if section_chunks:
        # Create an index for sections
        section_texts = [c['content'] for c in section_chunks]
        group_id = int(hashlib.md5(f"sections_{section_chunks[0]['filename']}".encode()).hexdigest(), 16) % 1000000
        
        print(f"Indexing {len(section_chunks)} section chunks with group ID {group_id}")
        embedder.create_index(group_id, section_texts, section_chunks)
    
    # Process paragraph chunks
    if paragraph_chunks:
        # Create an index for paragraphs
        paragraph_texts = [c['content'] for c in paragraph_chunks]
        group_id = int(hashlib.md5(f"paragraphs_{paragraph_chunks[0]['filename']}".encode()).hexdigest(), 16) % 1000000
        
        print(f"Indexing {len(paragraph_chunks)} paragraph chunks with group ID {group_id}")
        embedder.create_index(group_id, paragraph_texts, paragraph_chunks)
    
    # Process each chunk for entity extraction and URL crawling
    for chunk in tqdm(chunks, desc="Processing entities and URLs"):
        # Skip if no content
        if not chunk.get('content'):
            continue
        
        # Extract entities
        entities = embedder.extract_entities(chunk['content'])
        
        # Store chunk in Neo4j
        chunk_id = chunk['id']
        embedder.neo4j.store_chunk(chunk_id, chunk)
        
        # Process entities and add to Neo4j
        for entity_type, entity_names in entities.items():
            for entity_name in entity_names:
                # Create entity
                embedder.neo4j.create_entity(entity_type, entity_name)
                
                # Link entity to chunk
                embedder.neo4j.link_entity_to_chunk(entity_type, entity_name, chunk_id)
        
        # Extract and store URLs
        # Both normal URLs and those in <<url>> format will be extracted
        urls = embedder.extract_urls(chunk['content'])
        for url in urls:
            context = {
                'chunk_id': chunk_id,
                'filename': chunk['filename'],
                'title': chunk.get('title', ''),
                'level': chunk['level']
            }
            embedder.store_url(url, context)
    
    print(f"Indexed {len(chunks)} chunks with entity extraction and URL processing")
    # Return the formatted content of the first chunk for reference
    if chunks:
        return chunks[0]['content']

def ingest_document(group_id: int, texts: List[str], metadata: List[Dict] = None, 
                   api_key: str = None, base_path: str = "faiss_dbs") -> None:
    """
    Convenient function to ingest a document into FAISS.
    
    Args:
        group_id: Identifier for the group
        texts: List of text chunks to index
        metadata: Optional metadata for each text chunk
        api_key: Mistral API key
        base_path: Base directory to store FAISS indices
    """
    embedder = FaissEmbedder(api_key=api_key, base_path=base_path)
    
    # Format URLs in each text chunk before ingestion

    formatted_texts = [format_urls_in_content(extract_info_url(text)) for text in texts]
    
    embedder.create_index(group_id, formatted_texts, metadata)
    
def query_document(group_id: int, query: str, k: int = 5, 
                  api_key: str = None, base_path: str = "faiss_dbs") -> List[Dict]:
    """
    Convenient function to query a document from FAISS.
    
    Args:
        group_id: Identifier for the group
        query: Query text
        k: Number of results to return
        api_key: Mistral API key
        base_path: Base directory to store FAISS indices
        
    Returns:
        List of dictionaries with search results
    """
    embedder = FaissEmbedder(api_key=api_key, base_path=base_path)
    return embedder.search(group_id, query, k)

def section_faiss_search(query: str, k: int = 5, 
                       api_key: str = None, base_path: str = "faiss_dbs") -> List[Dict]:
    """
    Search for section-level chunks using FAISS.
    
    Args:
        query: Query text
        k: Number of results to return
        api_key: Mistral API key
        base_path: Base directory to store FAISS indices
        
    Returns:
        List of dictionaries with search results from sections across all groups
    """
    embedder = FaissEmbedder(api_key=api_key, base_path=base_path)
    
    # Get all available groups
    groups = embedder.get_all_groups()
    
    all_results = []
    for group_id in groups:
        try:
            # Load index and mapping for this group
            index, mapping = embedder.load_index(group_id)
            
            # Check if this index contains section-level chunks
            has_sections = any(m.get('level') == 'section' for m in mapping if isinstance(m, dict))
            
            if has_sections:
                # Embed query
                query_embedding = embedder.embed_texts([query])[0].reshape(1, -1)
                
                # Search
                distances, indices = index.search(query_embedding, k)
                
                # Filter and format results
                for i, idx in enumerate(indices[0]):
                    if idx < len(mapping):
                        result = mapping[idx].copy()
                        # Only include section-level chunks
                        if result.get('level') == 'section':
                            result["distance"] = float(distances[0][i])
                            result["group_id"] = group_id
                            all_results.append(result)
        except Exception as e:
            print(f"Error searching group {group_id} for sections: {e}")
    
    # Sort results by distance
    all_results.sort(key=lambda x: x.get('distance', float('inf')))
    
    # Return the top k results
    return all_results[:k]

def paragraph_faiss_search(query: str, k: int = 5, 
                         api_key: str = None, base_path: str = "faiss_dbs") -> List[Dict]:
    """
    Search for paragraph-level chunks using FAISS.
    
    Args:
        query: Query text
        k: Number of results to return
        api_key: Mistral API key
        base_path: Base directory to store FAISS indices
        
    Returns:
        List of dictionaries with search results from paragraphs across all groups
    """
    embedder = FaissEmbedder(api_key=api_key, base_path=base_path)
    
    # Get all available groups
    groups = embedder.get_all_groups()
    
    all_results = []
    for group_id in groups:
        try:
            # Load index and mapping for this group
            index, mapping = embedder.load_index(group_id)
            
            # Check if this index contains paragraph-level chunks
            has_paragraphs = any(m.get('level') == 'paragraph' for m in mapping if isinstance(m, dict))
            
            if has_paragraphs:
                # Embed query
                query_embedding = embedder.embed_texts([query])[0].reshape(1, -1)
                
                # Search
                distances, indices = index.search(query_embedding, k)
                
                # Filter and format results
                for i, idx in enumerate(indices[0]):
                    if idx < len(mapping):
                        result = mapping[idx].copy()
                        # Only include paragraph-level chunks
                        if result.get('level') == 'paragraph':
                            result["distance"] = float(distances[0][i])
                            result["group_id"] = group_id
                            all_results.append(result)
        except Exception as e:
            print(f"Error searching group {group_id} for paragraphs: {e}")
    
    # Sort results by distance
    all_results.sort(key=lambda x: x.get('distance', float('inf')))
    
    # Return the top k results
    return all_results[:k]

def retrieve_from_structured(query: str, db_path: str = None, 
                           api_key: str = None, base_path: str = "faiss_dbs") -> Dict:
    """
    Retrieve information from both structured data and FAISS indices.
    
    This function combines retrieval from:
    1. FAISS indices for unstructured text (sections and paragraphs)
    2. (If available) SQL database for structured data
    
    Args:
        query: Query text
        db_path: Path to SQLite database (if any)
        api_key: Mistral API key
        base_path: Base directory for FAISS indices
        
    Returns:
        Dictionary with combined search results
    """
    results = {
        'sections': [],
        'paragraphs': [],
        'structured_data': None,
        'entities': []
    }
    
    # Get sections and paragraphs from FAISS
    try:
        results['sections'] = section_faiss_search(query, k=3, api_key=api_key, base_path=base_path)
        results['paragraphs'] = paragraph_faiss_search(query, k=5, api_key=api_key, base_path=base_path)
    except Exception as e:
        print(f"Error retrieving from FAISS: {e}")
    
    # Create embedder to access Neo4j
    embedder = FaissEmbedder(api_key=api_key, base_path=base_path)
    
    # Extract entities from the query using spaCy
    if nlp:
        doc = nlp(query)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        results['entities'] = entities
        
        # If we have Neo4j connected, try to find related documents for each entity
        if embedder.neo4j.driver:
            try:
                for entity_type, entity_names in entities.items():
                    for entity_name in entity_names:
                        # Query Neo4j for documents related to this entity
                        with embedder.neo4j.driver.session() as session:
                            query = (
                                f"MATCH (e:{entity_type})-[:MENTIONED_IN]->(d:Document) "
                                "WHERE e.name = $entity_name "
                                "RETURN d.id, d.filename, d.title, d.level "
                                "LIMIT 5"
                            )
                            
                            records = session.run(query, entity_name=entity_name)
                            for record in records:
                                # Add this document to the results
                                doc_id = record["d.id"]
                                doc_data = {
                                    "id": doc_id,
                                    "filename": record["d.filename"],
                                    "title": record["d.title"],
                                    "level": record["d.level"],
                                    "matched_entity": entity_name,
                                    "entity_type": entity_type
                                }
                                
                                # Add to appropriate result list
                                if record["d.level"] == "section":
                                    if doc_data not in results['sections']:
                                        results['sections'].append(doc_data)
                                elif record["d.level"] == "paragraph":
                                    if doc_data not in results['paragraphs']:
                                        results['paragraphs'].append(doc_data)
            except Exception as e:
                print(f"Error querying Neo4j for entities: {e}")
    
    # Get structured data from SQLite if a database path is provided
    if db_path and os.path.exists(db_path):
        try:
            import sqlite3
            import pandas as pd
            
            # Connect to the database
            conn = sqlite3.connect(db_path)
            
            # Get list of tables
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            
            structured_results = {}
            
            # For each table, try to find relevant rows
            for table_name in table_names:
                if table_name.startswith('sqlite_') or table_name.startswith('_'):
                    continue  # Skip internal SQLite tables
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Simple keyword search in text columns
                text_columns = []
                for col in columns:
                    # Check if this column is likely to contain text
                    cursor.execute(f"SELECT typeof({col}) FROM {table_name} LIMIT 1;")
                    col_type = cursor.fetchone()
                    if col_type and col_type[0].lower() == 'text':
                        text_columns.append(col)
                
                # If we have text columns, search for the query keywords
                if text_columns:
                    # Create a WHERE clause for each text column
                    where_clauses = []
                    for col in text_columns:
                        for keyword in query.split():
                            if len(keyword) > 3:  # Only search for meaningful keywords
                                where_clauses.append(f"{col} LIKE '%{keyword}%'")
                    
                    if where_clauses:
                        sql = f"SELECT * FROM {table_name} WHERE {' OR '.join(where_clauses)} LIMIT 10"
                        
                        try:
                            # Execute the query
                            df = pd.read_sql(sql, conn)
                            
                            if not df.empty:
                                structured_results[table_name] = df.to_dict('records')
                        except Exception as e:
                            print(f"Error querying table {table_name}: {e}")
            
            conn.close()
            results['structured_data'] = structured_results
            
        except ImportError:
            print("Could not import sqlite3 or pandas, skipping structured data retrieval")
        except Exception as e:
            print(f"Error retrieving from structured data: {e}")
    
    return results 