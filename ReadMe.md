# Document RAG Assistant

A sophisticated document processing and question answering system powered by Retrieval Augmented Generation (RAG). This application allows users to upload, process, and query various document types with natural language questions.

## Features

### Document Processing
- **Multi-format Support**:
  - Text documents: PDF, TXT, DOCX, PPTX
  - Image files: JPG, PNG (with OCR)
  - Structured data: CSV, XLSX, JSON
- **Intelligent Document Segmentation**:
  - Section-level chunking for context preservation
  - Paragraph-level chunking for detailed answers
  - Automatic detection of headings and subheadings
- **Advanced Document Analysis**:
  - Image extraction and OCR processing
  - Hyperlink detection and processing
  - Table recognition

### Retrieval and Answering
- **Smart Query Routing**:
  - ENTITY: Entity-focused semantic search
  - DATA: Structured data queries via SQL or pandas
  - GENERAL: Section-level semantic search
  - DETAILED: Paragraph-level semantic search
  - VAGUE: Hybrid search approach
- **Contextual Relevance**:
  - Vector similarity search via FAISS
  - Context-aware answer generation
  - Multi-source evidence combination

### User Experience
- **Web Interface**:
  - Upload and manage documents
  - Ask questions in natural language
  - View and manage processing status
- **Deployment Ready**:
  - Local development setup
  - Azure cloud deployment
  - GitHub Actions integration

## Architecture

### System Components

```
┌────────────────────────────────┐
│        Web Application         │
│  ┌──────────────────────────┐  │
│  │    Flask Web Server     │  │
│  └──────────────────────────┘  │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│     Document Processing        │
│  ┌──────────┐  ┌────────────┐  │
│  │ Document │  │ Structured │  │
│  │ Converter│  │    Data    │  │
│  └────┬─────┘  └─────┬──────┘  │
│       │              │         │
│  ┌────▼─────────────▼──────┐   │
│  │      Chunking &         │   │
│  │   Embedding Pipeline    │   │
│  └───────────┬─────────────┘   │
└──────────────┼─────────────────┘
               │
               ▼
┌────────────────────────────────┐
│      Storage & Indexing        │
│  ┌─────────┐  ┌─────────────┐  │
│  │  FAISS  │  │  SQLite DB  │  │
│  │ Indices │  │             │  │
│  └─────────┘  └─────────────┘  │
└───────────────┬─────────────────┘
                │
                ▼
┌────────────────────────────────┐
│       Query Processing         │
│  ┌───────────────────────────┐ │
│  │ Classification Pipeline   │ │
│  └───────────┬───────────────┘ │
│              │                 │
│  ┌───────────▼───────────────┐ │
│  │   Retrieval Pipeline      │ │
│  └───────────┬───────────────┘ │
│              │                 │
│  ┌───────────▼───────────────┐ │
│  │   Answer Generation       │ │
│  └───────────────────────────┘ │
└────────────────────────────────┘
```

### Data Flow

1. **Document Ingestion**:
   - Documents are uploaded through the web interface
   - Each document receives a unique timestamp+UUID identifier
   - Documents are processed according to their type:
     - Text documents (PDF, DOCX, TXT) → text extraction
     - Image files → OCR processing
     - Structured data → DataFrame/SQL conversion

2. **Document Transformation**:
   - Conversion to markdown format for standardization
   - Intelligent chunking into sections and paragraphs
   - Entity extraction (using spaCy)
   - URL detection and content extraction
   - Structured data schema analysis

3. **Embedding & Indexing**:
   - Text chunks embedded with Voyage AI embeddings
   - Vector indices created and stored with FAISS
   - Structured data indexed in SQLite
   - Optional: Entities stored in Neo4j graph database

4. **Query Processing**:
   - Natural language query classification
   - Route-specific retrieval strategies
   - Contextually relevant information retrieval
   - Answer synthesis with Claude 3.7 Sonnet

## RAG Approach

The system implements a sophisticated Retrieval Augmented Generation approach:

### 1. Document Processing Pipeline

- **Chunking Strategy**: 
  - Section-level chunks preserve broader context
  - Paragraph-level chunks provide detailed information
  - Chunk size optimization based on document type
  - Metadata preservation for context reconstruction

- **Embedding Generation**:
  - Voyage AI embeddings (voyage-3-lite model)
  - Embedding caching for frequently used queries
  - Batch processing for efficiency
  - Error handling with exponential backoff

### 2. Query Processing Pipeline

- **Query Classification**:
  - `ENTITY`: Entity-focused queries (people, organizations, etc.)
  - `DATA`: Queries about structured data or statistics
  - `GENERAL`: Broad informational queries
  - `DETAILED`: Specific information lookup
  - `VAGUE`: Ambiguous queries requiring hybrid approach

- **Contextual Retrieval**:
  - Vector similarity search using FAISS indices
   - Query embedding reuse via caching
   - Relevance scoring and ranking
   - Multi-source evidence combination

### 3. Answer Generation

- **Context Assembly**:
  - Relevant chunks assembled with metadata
  - Source attribution preservation
  - Context ordering by relevance
  - Context size optimization for model input

- **LLM Integration**:
  - Claude 3.7 Sonnet for answer synthesis
  - Prompt engineering for accurate responses
  - Fallback mechanisms for low-confidence answers
  - Input description for context awareness

## Setup and Installation

### Prerequisites

- Python 3.11+
- API keys for:
  - Anthropic Claude (required)
  - Voyage AI (required for embeddings)
  - Google Gemini (optional, for document processing)
  - Mistral AI (optional)

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/document-rag-assistant.git
cd document-rag-assistant
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

4. **Set environment variables**

```bash
# Required
export ANTHROPIC_API_KEY=your_anthropic_api_key
export VOYAGE_API_KEY=your_voyage_api_key

# Optional
export GEMINI_API_KEY=your_gemini_api_key  
export MISTRAL_API_KEY=your_mistral_api_key
```

5. **Run the application**

```bash
python app.py
```

6. **Access the application**

Open your browser and navigate to `http://localhost:5000`.

### Docker Deployment

1. **Build the Docker image**

```bash
docker build -t document-rag-assistant .
```

2. **Run the container**

```bash
docker run -p 5000:5000 \
  -e ANTHROPIC_API_KEY=your_anthropic_api_key \
  -e VOYAGE_API_KEY=your_voyage_api_key \
  -e GEMINI_API_KEY=your_gemini_api_key \
  document-rag-assistant
```

## Usage

### Document Upload

1. Navigate to the web interface at `http://localhost:5000`
2. Click the "Upload Documents" button
3. Select one or more files to upload (PDF, DOCX, TXT, JPG, PNG, CSV, XLSX, JSON)
4. Wait for processing to complete

### Asking Questions

1. Enter your question in the text field
2. Click "Ask" or press Enter
3. View the answer and related document context

### Command Line Interface

For advanced users, a command-line interface is available:

```bash
python main.py
```

This provides options for:
- Document ingestion
- Structured data loading
- Direct question answering
- SQL query execution

## Integration with External Services

### Anthropic Claude

The system uses Claude 3.7 Sonnet for:
- Answer generation from retrieved context
- Query classification
- Document summarization (optional)

### Voyage AI

Used for generating document and query embeddings:
- High-quality semantic vectors
- Consistency between document and query embeddings
- Batch processing capabilities

### Optional: Neo4j

For entity-centric applications, Neo4j can be enabled to:
- Store and query named entities
- Build knowledge graphs from documents
- Support entity-centric queries

### Optional: Google Gemini

Used for:
- OCR on images within documents
- Document summarization
- Visual content analysis

## Deployment Options

### Azure Web App Deployment

1. **Create Azure Web App**

```bash
az webapp create --resource-group YourResourceGroup --plan YourAppServicePlan --name YourAppName --runtime "PYTHON:3.11"
```

2. **Configure environment variables**

```bash
az webapp config appsettings set --resource-group YourResourceGroup --name YourAppName --settings ANTHROPIC_API_KEY=your_key VOYAGE_API_KEY=your_key
```

3. **Deploy using GitHub Actions**

Set up a GitHub Actions workflow using `.github/workflows/azure-deploy.yaml`

### Manual Deployment

1. Zip your application files
2. Upload to your hosting provider
3. Configure environment variables
4. Start the application

## Performance Considerations

- **Memory Usage**: FAISS indices can consume substantial memory for large document collections
- **API Costs**: The system makes API calls to external services, which have associated costs
- **Processing Time**: Document ingestion includes embedding generation which can take time
- **Scaling**: For large-scale deployments, consider:
  - Distributed FAISS indices
  - Database sharding
  - Azure autoscaling setup

## Advanced Configuration

### Chunking Strategies

Edit the `faiss_embedder.py` file to adjust:
- `DEFAULT_CHUNK_SIZE`: Controls paragraph chunk size
- `MAX_TOKENS_BEFORE_SECTION_SPLIT`: Controls when to split sections

### Embedding Models

The system uses Voyage AI by default, but can be configured to use other embedding providers by modifying the `FaissEmbedder` class.

### Query Processing Pipeline

The query processing pipeline in `main.py` can be customized by modifying:
- Classification logic
- Retrieval strategies
- Answer generation prompts

## License

MIT