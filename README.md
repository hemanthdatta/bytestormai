# Document RAG Assistant

A web application that allows users to upload documents and ask questions about them using Retrieval Augmented Generation (RAG).

## Features

- Upload multiple document types (PDF, TXT, DOCX, PPTX, JPG, PNG)
- Upload structured data files (CSV, XLSX, JSON)
- Ask questions about your documents in natural language
- Intelligent retrieval and response generation
- Modern, ChatGPT-like user interface

## Local Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```
   export MISTRAL_API_KEY=your_mistral_api_key
   export GEMINI_API_KEY=your_gemini_api_key  # Optional, used for document processing
   ```
4. Run the application:
   ```
   python app.py
   ```
5. Open http://localhost:5000 in your browser

## Deploying to Azure

### Option 1: Manual Deployment

1. Create an Azure Web App with Python runtime
2. Set up the following application settings in the Azure portal:
   - MISTRAL_API_KEY: Your Mistral API key
   - GEMINI_API_KEY: Your Gemini API key (optional)
3. Deploy the code using Azure CLI, Visual Studio Code, or GitHub Actions

### Option 2: GitHub Actions Deployment

1. Fork this repository
2. Create an Azure Web App
3. Generate a publish profile from the Azure portal
4. Add the publish profile as a GitHub secret named `AZURE_WEBAPP_PUBLISH_PROFILE`
5. Push to the main branch to trigger the deployment workflow

## Architecture

The application uses a Flask web server with the following components:
- Frontend: HTML/CSS/JS with a ChatGPT-like interface
- Backend: Python-based RAG pipeline that processes documents and answers questions
- File processing: Support for various document types
- Azure deployment: Configuration for hosting on Azure Web App

## License

MIT 