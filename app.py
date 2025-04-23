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
    return render_template('index.html')

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
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_paths.append(filepath)
            
            # Categorize files
            ext = filename.rsplit('.', 1)[1].lower()
            if ext in ['csv', 'xlsx', 'json']:
                structured_paths.append(filepath)
            else:
                document_paths.append(filepath)
    
    # Process files with the main module
    if document_paths:
        main.ingest_documents(document_paths)
    
    if structured_paths:
        main.ingest_structured(structured_paths)
    
    return jsonify({
        'message': f'Successfully uploaded {len(uploaded_paths)} files',
        'uploads': uploaded_paths
    })

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    result = main.pipeline.invoke({'question': question})
    
    return jsonify({
        'answer': result.get('answer', 'Sorry, I could not find an answer.'),
        'question': question
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 