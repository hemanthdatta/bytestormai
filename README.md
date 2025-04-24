# Document RAG Assistant

A web application that allows users to upload documents and ask questions about them using Retrieval Augmented Generation (RAG).

## Features

- Upload and process multiple document types:
  - Text documents: PDF, TXT, DOCX, PPTX
  - Image files: JPG, PNG
  - Structured data: CSV, XLSX, JSON
- Ask questions about your documents in natural language
- Accurate answers powered by intelligent retrieval and generation

## Local Setup

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables**:
   ```bash
   export MISTRAL_API_KEY=your_mistral_api_key
   export GEMINI_API_KEY=your_gemini_api_key            # Optional, used for document processing
   export ANTHROPIC_API_KEY=your_anthropic_api_key      # Required for Claude model-based answering
   export VOYAGE_API_KEY=your_voyage_api_key            # Required for Voyage embedding model
   ```
4. **Run the application**:
   ```bash
   python app.py
   ```
5. **Access the app**:  
   After the server starts, it will display a `localhost` URL in the terminal (usually `http://localhost:5000`). Open it in your browser to use the assistant.

## Deploying to Azure

### Option 1: Manual Deployment

1. Create an Azure Web App with Python runtime
2. Set the following application settings in the Azure portal:
   - `MISTRAL_API_KEY`: Your Mistral API key
   - `GEMINI_API_KEY`: Your Gemini API key (optional)
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - `VOYAGE_API_KEY`: Your Voyage API key
3. Deploy the code using Azure CLI, Visual Studio Code, or GitHub Actions

### Option 2: GitHub Actions Deployment

1. Fork this repository
2. Create an Azure Web App
3. Generate a publish profile from the Azure portal
4. Add the publish profile as a GitHub secret named `AZURE_WEBAPP_PUBLISH_PROFILE`
5. Push to the main branch to trigger the deployment workflow

## Architecture

The application uses a Flask web server with the following components:
- **Frontend**: HTML/CSS/JS with a ChatGPT-style interface
- **Backend**: Python-based RAG pipeline that processes documents and answers questions
- **File handling**: Supports diverse document types, text and structured
- **Deployment-ready**: Easily hostable on Azure Web App with GitHub Actions

## Deployment

This project is deployed as a live website through **Azure Web App**, allowing users to interact with the Document RAG Assistant directly from their browser.

## Pipeline Overview

When documents are ingested, they are first converted into chunks and then stored in different embedding stores:

- **Text Embeddings**: Chunks of the document are embedded using the Voyage API and indexed in FAISS for semantic search.
- **Structured Data**: Tables and spreadsheets are loaded into pandas DataFrames and saved in an SQLite database for SQL-based retrieval.
- **Entity Graph**: Named entities are extracted via spaCy and stored in Neo4j as a graph for entity-centric queries.

At query time, the following steps occur:

1. **Classification**: The user’s question is classified into one of five routes (ENTITY, DATA, GENERAL, DETAILED, VAGUE).
2. **Retrieval**: Based on the route:
   - ENTITY → section-level FAISS search
   - DATA   → SQL query or pandas-agent fallback
   - GENERAL/DETAILED → section/paragraph FAISS search
3. **Link Handling**: Any URLs embedded in documents are summarized and embedded so their context contributes to retrieval; if a link is relevant, its full text is fetched and passed to the model.
4. **Answer Generation**: The retrieved context is combined into a prompt and sent to the Claude model for final answer synthesis.

## Limitations

- **Latency**: Processing large documents (chunking, embedding, entity extraction) can be time-consuming.
- **URL Summarization**: Fetching and summarizing external links adds additional delay.
- **API Dependencies**: Multiple external API calls (Voyage embeddings, Anthropic completions) are subject to rate limits and network latency.
- **English Only**: The system currently supports English-language documents and queries.
- **Resource Usage**: Embeddings and graph storage can consume significant compute and memory for large corpora.

## Purpose

This project is focused on accurately answering questions from a wide range of document types. It leverages retrieval-augmented generation (RAG) techniques and multiple language models to ensure relevant and concise answers.

## License

MIT

