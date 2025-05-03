from flask import Flask, render_template, request, jsonify, session
import os
import uuid
from werkzeug.utils import secure_filename
import main

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'pptx', 'jpg', 'png', 'csv', 'xlsx', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Get list of uploaded files to display in UI
    uploaded_files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                filesize = os.path.getsize(filepath)
                uploaded_files.append({
                    'name': filename,
                    'path': filepath,
                    'size': filesize
                })
    
    return render_template('index.html', uploaded_files=uploaded_files)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Create session if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Process uploaded files
    uploaded_paths = []
    document_paths = []
    structured_paths = []
    file_details = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filesize = os.path.getsize(filepath)
            
            uploaded_paths.append(filepath)
            
            # Categorize files
            ext = filename.rsplit('.', 1)[1].lower()
            if ext in ['csv', 'xlsx', 'json']:
                structured_paths.append(filepath)
            else:
                document_paths.append(filepath)
                
            # Add file details for UI
            file_details.append({
                'name': filename,
                'path': filepath,
                'size': filesize,
                'type': 'structured' if ext in ['csv', 'xlsx', 'json'] else 'document'
            })
    
    # Process files with the main module
    if document_paths:
        try:
            main.minimal_ingest_documents(document_paths)
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            return jsonify({'error': f'Document processing error: {str(e)}'}), 500
    
    if structured_paths:
        try:
            main.ingest_structured(structured_paths)
        except Exception as e:
            print(f"Error ingesting structured data: {e}")
            return jsonify({'error': f'Structured data processing error: {str(e)}'}), 500
    
    return jsonify({
        'message': f'Successfully uploaded {len(uploaded_paths)} files',
        'uploads': uploaded_paths,
        'file_details': file_details
    })

@app.route('/delete-file', methods=['POST'])
def delete_file():
    data = request.json
    if not data or 'filepath' not in data:
        return jsonify({'error': 'No filepath provided'}), 400
    
    filepath = data['filepath']
    filename = os.path.basename(filepath)
    
    # Security check to ensure we only delete files in the upload folder
    if not os.path.normpath(filepath).startswith(os.path.normpath(app.config['UPLOAD_FOLDER'])):
        return jsonify({'error': 'Invalid filepath'}), 403
    
    try:
        # Delete the file
        if os.path.exists(filepath):
            os.remove(filepath)
            
            # Clean up from memory
            try:
                main.remove_from_memory(filepath)
            except Exception as e:
                print(f"Error removing from memory: {e}")
            
            return jsonify({
                'message': f'Successfully deleted {filename}',
                'deleted': filepath
            })
        else:
            return jsonify({'error': f'File {filename} not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error deleting file: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Clear the query embedding cache to ensure fresh embeddings for new queries
        import faiss_embedder
        faiss_embedder.clear_query_cache()
        
        question = data['question']
        result = main.pipeline.invoke({'question': question})
        
        return jsonify({
            'answer': result.get('answer', 'Sorry, I could not find an answer.'),
            'question': question
        })
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 