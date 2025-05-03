import os
import anthropic
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
from anthropic import Anthropic
import voyageai
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
anthropic_api_key = "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"
voyage_api_key    = "pa-RofV7BIpsxPQxTiGqbuMA18pw6tZYVSl-1h7zwrYoGJ"

# Constants
DEFAULT_CHUNK_SIZE = 500
MAX_TOKENS_BEFORE_SECTION_SPLIT = 5000
URL_REGEX = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
MD_URL_REGEX = r'<<(https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+)>>'
SIMPLIFIED_URL = r'<<.*?>>'
DB_PATH = "url_cache.db"

# Add a global query embedding cache
QUERY_EMBEDDING_CACHE = {}

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Download spaCy model: python -m spacy download en_core_web_sm")
    nlp = None


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def extract_visible_text(url: str) -> str:
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'lxml')
        for tag in soup(['script', 'style', 'head', 'meta', 'noscript']):
            tag.decompose()
        texts = soup.find_all(string=True)
        visible = filter(tag_visible, texts)
        return '\n'.join(t.strip() for t in visible if t.strip())
    except Exception as e:
        return f"Error: {e}"


class Summarizer:
    def __init__(self,
                 model: str = 'claude-3-haiku-20240307',
                 max_retries: int = 5,
                 retry_delay: float = 0.01):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.system_prompt = "Summarise the given context into a 3-4 line paragraph."

    def summarise_text(self, text: str) -> str:
        # Use top-level 'system' parameter instead of role in messages
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=200
                )
                print(f"Summarizer response: {response.content[0].text}")
                return str(response.content[0])
            except Exception as e:
                print(f"Summarizer retry {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return "[Summary failed]"
    
summarizer = Summarizer()

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
    
    def clear_all_data(self):
        """
        Clear all data from the Neo4j database.
        This removes all nodes and relationships.
        
        Returns:
            bool: Success status
        """
        if not self.driver:
            print("No Neo4j connection available.")
            return False
            
        try:
            with self.driver.session() as session:
                # Delete all relationships first
                session.run("MATCH ()-[r]-() DELETE r")
                # Then delete all nodes
                session.run("MATCH (n) DELETE n")
                print("All Neo4j data has been cleared")
                return True
        except Exception as e:
            print(f"Failed to clear Neo4j data: {e}")
            return False
    
    def clear_document_data(self, filename):
        """
        Clear data related to a specific document from the Neo4j database.
        This removes Document nodes matching the filename and related relationships.
        
        Args:
            filename: Name of the document file to clear
            
        Returns:
            bool: Success status
        """
        if not self.driver:
            print("No Neo4j connection available.")
            return False
            
        try:
            with self.driver.session() as session:
                # Find Document nodes with matching filename
                result = session.run(
                    "MATCH (d:Document) WHERE d.filename = $filename RETURN count(d) as count",
                    filename=filename
                )
                count = result.single()["count"]
                
                if count == 0:
                    print(f"No documents found with filename: {filename}")
                    return False
                
                # Delete relationships connected to document nodes
                session.run(
                    "MATCH (d:Document)-[r]-() WHERE d.filename = $filename DELETE r",
                    filename=filename
                )
                
                # Delete document nodes
                result = session.run(
                    "MATCH (d:Document) WHERE d.filename = $filename DELETE d RETURN count(d) as deleted",
                    filename=filename
                )
                deleted = result.single()["deleted"]
                
                print(f"Cleared data for document '{filename}': {deleted} nodes removed")
                return True
        except Exception as e:
            print(f"Failed to clear document data: {e}")
            return False
    
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
    def __init__(
        self,
        api_key: str,
        base_path: str = "faiss_dbs",
        model: str = "voyage-3-lite",
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None
    ):
        self.base_path = base_path
        self.emb_model = model
        os.makedirs(base_path, exist_ok=True)
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.indices: Dict[int, faiss.Index] = {}
        self.mappings: Dict[int, List[dict]] = {}
        
        # Only initialize Neo4j connection if valid URI is provided
        if neo4j_uri and neo4j_uri.strip():
            self.neo4j = Neo4jConnector(neo4j_uri, neo4j_user or '', neo4j_password or '')
        else:
            self.neo4j = None
            print("Neo4j connection not initialized: No valid URI provided")
            
        self.url_store_path = os.path.join(base_path, "url_store")
        os.makedirs(self.url_store_path, exist_ok=True)
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'neo4j') and self.neo4j:
            self.neo4j.close()

    def embed_texts(self, texts: List[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
        """
        Embed a list of texts using the Voyage AI API.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed at once
            is_query: Whether this is a query text (for caching)
            
        Returns:
            Array of embeddings
        """
        # Check cache for query texts
        if is_query and len(texts) == 1:
            query_text = texts[0]
            cache_key = f"{query_text}_{self.emb_model}"
            if cache_key in QUERY_EMBEDDING_CACHE:
                print(f"Using cached embedding for query: {query_text[:30]}...")
                return QUERY_EMBEDDING_CACHE[cache_key]
        
        embeds: List[np.ndarray] = []
        import math
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i + batch_size]
            for attempt in range(5):
                try:
                    resp = self.voyage_client.embed(
                        texts=batch,
                        model=self.emb_model,
                        input_type="document"
                    )
                    embeds.extend(resp.embeddings)
                    break
                except Exception as e:
                    # Use exponential backoff
                    backoff_time = 0.1 * math.pow(2, attempt)
                    print(f"Embedding retry {attempt+1}/5 failed: {e}. Waiting {backoff_time:.2f}s")
                    time.sleep(backoff_time)
                    if attempt == 4:
                        print(f"Failed to embed batch after 5 retries")
                        # Return dummy embeddings to prevent complete failure
                        dummy_size = 1024  # Voyage embedding size
                        embeds.extend([np.zeros(dummy_size) for _ in batch])
        
        result = np.array(embeds, dtype=np.float32)
        
        # Cache the result for query texts
        if is_query and len(texts) == 1:
            query_text = texts[0]
            cache_key = f"{query_text}_{self.emb_model}"
            QUERY_EMBEDDING_CACHE[cache_key] = result
        
        return result

    def create_index(self, gid: int, texts: List[str], meta: List[dict] = None) -> None:
        if not texts:
            print(f"No texts for group {gid}")
            return
        group_dir = os.path.join(self.base_path, f"group_{gid}")
        os.makedirs(group_dir, exist_ok=True)
        embeddings = self.embed_texts(texts)
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32) if len(texts) > 10000 else faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(group_dir, "index.faiss"))
        mapping = []
        for i, text in enumerate(texts):
            entry = (meta[i] if meta else {}).copy()
            entry.update({"id": i, "text": text})
            mapping.append(entry)
        with open(os.path.join(group_dir, "mapping.pkl"), 'wb') as f:
            pickle.dump(mapping, f)
        self.indices[gid] = index
        self.mappings[gid] = mapping

    def load_index(self, gid: int) -> Tuple[faiss.Index, List[dict]]:
        if gid in self.indices:
            return self.indices[gid], self.mappings[gid]
        group_dir = os.path.join(self.base_path, f"group_{gid}")
        index = faiss.read_index(os.path.join(group_dir, "index.faiss"))
        with open(os.path.join(group_dir, "mapping.pkl"), 'rb') as f:
            mapping = pickle.load(f)
        self.indices[gid] = index
        self.mappings[gid] = mapping
        return index, mapping

    def search(self, gid: int, query: str, k: int = 5) -> List[dict]:
        index, mapping = self.load_index(gid)
        query_emb = self.embed_texts([query], is_query=True)[0].reshape(1, -1)
        dists, ids = index.search(query_emb, k)
        results: List[dict] = []
        for dist, idx in zip(dists[0], ids[0]):
            if idx < len(mapping):
                item = mapping[idx].copy()
                item['distance'] = float(dist)
                results.append(item)
        return results

    def delete_index(self, gid: int) -> None:
        group_dir = os.path.join(self.base_path, f"group_{gid}")
        if os.path.isdir(group_dir):
            for fn in os.listdir(group_dir):
                os.remove(os.path.join(group_dir, fn))
            os.rmdir(group_dir)
        self.indices.pop(gid, None)
        self.mappings.pop(gid, None)

    def get_all_groups(self) -> List[int]:
        return [int(d.split('_')[1]) for d in os.listdir(self.base_path) if d.startswith('group_')]
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
                neo4j_uri: str = None,
                neo4j_user: str = "neo4j",
                neo4j_password: str = "CMB2JFluGdYmo5kNG2x7qeAA8krSJK32GTgAJogmYdA",
                process_urls: bool = True,
                process_entities: bool = True,
                minimal_processing: bool = False) -> None:
    """
    Index a list of chunks using FAISS, spaCy for entity extraction, and Neo4j.
    
    Args:
        chunks: List of chunk dictionaries
        api_key: API key for embeddings
        base_path: Base directory for FAISS indices
        neo4j_uri: Neo4j connection URI (None to disable Neo4j)
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        process_urls: Whether to process URLs (can be disabled for speed)
        process_entities: Whether to extract entities (can be disabled for speed)
        minimal_processing: If True, only creates FAISS indexes without entity/URL processing
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
    
    # For small documents (less than 20 chunks), disable entity processing unless explicitly requested
    is_small_document = len(chunks) < 20
    if is_small_document and process_entities:
        print(f"Small document detected ({len(chunks)} chunks). Consider using minimal_processing=True for faster indexing.")
    
    # Skip URL processing if disabled or in minimal processing mode
    if process_urls and not minimal_processing:
        from concurrent.futures import ThreadPoolExecutor
        
        def process_chunk_urls(chunk):
            if 'content' in chunk:
                chunk['content'] = extract_info_url(chunk['content'])
                chunk['content'] = format_urls_in_content(chunk['content'])
            return chunk
        
        # Process URLs in parallel
        with ThreadPoolExecutor(max_workers=min(50, len(chunks))) as executor:
            chunks = list(executor.map(process_chunk_urls, chunks))
    
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
    
    # Skip entity extraction and Neo4j operations if in minimal processing mode
    if minimal_processing:
        print("Minimal processing enabled - skipping entity extraction and Neo4j operations")
        return
    
    # Skip entity extraction if disabled or document is small and auto-optimization is enabled
    if not process_entities or (is_small_document and not process_entities):
        print("Entity extraction disabled - skipping entity and Neo4j processing")
        return
        
    # Process entities in batches
    from concurrent.futures import ThreadPoolExecutor
    batch_size = 64  # Increased batch size for performance
    
    # Define a function to process a single chunk
    def process_chunk(chunk):
        if not chunk.get('content'):
            return None
        
        # Extract entities (if enabled)
        entities = embedder.extract_entities(chunk['content']) if process_entities else {}
        
        # Extract URLs (if enabled)
        urls = embedder.extract_urls(chunk['content']) if process_urls else []
        
        return {
            'chunk_id': chunk['id'],
            'entities': entities,
            'urls': urls,
            'chunk': chunk
        }
    
    # Process chunks in batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing entities and URLs"):
        batch = chunks[i:i+batch_size]
        
        # Process entities and URLs in parallel
        with ThreadPoolExecutor(max_workers=min(batch_size, len(batch))) as executor:
            results = list(executor.map(process_chunk, batch))
        
        # Filter out None results
        results = [r for r in results if r]
        
        # Skip Neo4j operations if no results
        if not results:
            continue
            
        # Prepare bulk operations
        all_chunk_ids = []
        all_chunks = []
        all_entities = []
        all_urls = []
        
        # Collect all data for bulk operations
        for result in results:
            chunk_id = result['chunk_id']
            chunk = result['chunk']
            
            all_chunk_ids.append(chunk_id)
            all_chunks.append(chunk)
            
            for entity_type, entity_names in result['entities'].items():
                for entity_name in entity_names:
                    all_entities.append((entity_type, entity_name, chunk_id))
            
            if process_urls:
                for url in result['urls']:
                    context = {
                        'chunk_id': chunk_id,
                        'filename': chunk['filename'],
                        'title': chunk.get('title', ''),
                        'level': chunk['level']
                    }
                    all_urls.append((url, context))
        
        # Perform Neo4j operations in bulk where possible
        try:
            # Only perform Neo4j operations if Neo4j is connected
            if embedder.neo4j and embedder.neo4j.driver:
                # Store chunks in Neo4j - still one at a time but in a batch
                for chunk_id, chunk in zip(all_chunk_ids, all_chunks):
                    embedder.neo4j.store_chunk(chunk_id, chunk)
                
                # Process entities using batch operations where possible
                if embedder.neo4j.driver and all_entities:
                    with embedder.neo4j.driver.session() as session:
                        # First create all entities in a single query
                        entity_types = list(set(e[0] for e in all_entities))
                        for entity_type in entity_types:
                            entities_of_type = [e[1] for e in all_entities if e[0] == entity_type]
                            if entities_of_type:
                                # Create entities of this type
                                query = (
                                    f"UNWIND $entities AS entity "
                                    f"MERGE (e:{entity_type} {{name: entity, id: apoc.util.md5([entity, '{entity_type}'])}})"
                                    f" RETURN count(e) as count"
                                )
                                try:
                                    session.run(query, entities=list(set(entities_of_type)))
                                except Exception as e:
                                    print(f"Error creating entities: {e}")
                                    # Fall back to individual entity creation
                                    for entity_name in entities_of_type:
                                        embedder.neo4j.create_entity(entity_type, entity_name)
                        
                        # Then link entities to chunks
                        for entity_type, entity_name, chunk_id in all_entities:
                            embedder.neo4j.link_entity_to_chunk(entity_type, entity_name, chunk_id)
            
            # Store URLs
            if process_urls and all_urls:
                for url, context in all_urls:
                    embedder.store_url(url, context)
        except Exception as e:
            print(f"Error performing Neo4j operations: {e}")
    
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
    
    # Embed query once instead of for each group - mark as query for caching
    query_embedding = embedder.embed_texts([query], is_query=True)[0].reshape(1, -1)
    
    all_results = []
    for group_id in groups:
        try:
            # Load index and mapping for this group
            index, mapping = embedder.load_index(group_id)
            
            # Check if this index contains section-level chunks
            has_sections = any(m.get('level') == 'section' for m in mapping if isinstance(m, dict))
            
            if has_sections:
                # Use the pre-computed query embedding
                
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
    
    # Embed query once instead of for each group - mark as query for caching
    query_embedding = embedder.embed_texts([query], is_query=True)[0].reshape(1, -1)
    
    all_results = []
    for group_id in groups:
        try:
            # Load index and mapping for this group
            index, mapping = embedder.load_index(group_id)
            
            # Check if this index contains paragraph-level chunks
            has_paragraphs = any(m.get('level') == 'paragraph' for m in mapping if isinstance(m, dict))
            
            if has_paragraphs:
                # Use the pre-computed query embedding
                
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
    
    # Create embedder once to reuse for all operations
    embedder = FaissEmbedder(api_key=api_key, base_path=base_path)
    
    # Get sections and paragraphs from FAISS
    try:
        # Get all available groups
        groups = embedder.get_all_groups()
        
        # Embed query once for all searches - mark as query for caching
        query_embedding = embedder.embed_texts([query], is_query=True)[0].reshape(1, -1)
        
        # Search for sections
        section_results = []
        for group_id in groups:
            try:
                # Load index and mapping for this group
                index, mapping = embedder.load_index(group_id)
                
                # Check if this index contains section-level chunks
                has_sections = any(m.get('level') == 'section' for m in mapping if isinstance(m, dict))
                
                if has_sections:
                    # Search
                    distances, indices = index.search(query_embedding, 3)  # k=3 for sections
                    
                    # Filter and format results
                    for i, idx in enumerate(indices[0]):
                        if idx < len(mapping):
                            result = mapping[idx].copy()
                            # Only include section-level chunks
                            if result.get('level') == 'section':
                                result["distance"] = float(distances[0][i])
                                result["group_id"] = group_id
                                section_results.append(result)
            except Exception as e:
                print(f"Error searching group {group_id} for sections: {e}")
        
        # Sort section results by distance
        section_results.sort(key=lambda x: x.get('distance', float('inf')))
        results['sections'] = section_results[:3]  # Top 3 sections
        
        # Search for paragraphs
        paragraph_results = []
        for group_id in groups:
            try:
                # Load index and mapping for this group
                index, mapping = embedder.load_index(group_id)
                
                # Check if this index contains paragraph-level chunks
                has_paragraphs = any(m.get('level') == 'paragraph' for m in mapping if isinstance(m, dict))
                
                if has_paragraphs:
                    # Search
                    distances, indices = index.search(query_embedding, 5)  # k=5 for paragraphs
                    
                    # Filter and format results
                    for i, idx in enumerate(indices[0]):
                        if idx < len(mapping):
                            result = mapping[idx].copy()
                            # Only include paragraph-level chunks
                            if result.get('level') == 'paragraph':
                                result["distance"] = float(distances[0][i])
                                result["group_id"] = group_id
                                paragraph_results.append(result)
            except Exception as e:
                print(f"Error searching group {group_id} for paragraphs: {e}")
        
        # Sort paragraph results by distance
        paragraph_results.sort(key=lambda x: x.get('distance', float('inf')))
        results['paragraphs'] = paragraph_results[:5]  # Top 5 paragraphs
        
    except Exception as e:
        print(f"Error retrieving from FAISS: {e}")
    
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
        if embedder.neo4j and embedder.neo4j.driver:
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
                            
                            # Process results
                            for record in records:
                                doc_info = {
                                    'filename': record['d.filename'],
                                    'title': record['d.title'],
                                    'level': record['d.level'],
                                    'entity_match': entity_name,
                                    'entity_type': entity_type,
                                    'source': 'neo4j'
                                }
                                
                                if doc_info not in results['entities']:
                                    results['entities'].append(doc_info)
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

def clear_query_cache():
    """Clear the query embedding cache."""
    global QUERY_EMBEDDING_CACHE
    old_size = len(QUERY_EMBEDDING_CACHE)
    QUERY_EMBEDDING_CACHE.clear()
    print(f"Cleared query cache: removed {old_size} cached embeddings") 