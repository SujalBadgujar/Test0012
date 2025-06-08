from flask import Flask, render_template, request, redirect, url_for, flash
import os
import zipfile
import shutil
from werkzeug.utils import secure_filename
from utils.text_processor import TextProcessor
from utils.pdf_handler import PDFHandler
from utils.similarity import calculate_similarity_matrix
from config import Config

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Hugging Face model for AI detection
try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
    model.eval()
    print("Hugging Face AI Detector model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        flash('No files uploaded')
        return redirect(url_for('index'))

    files = request.files.getlist('files')

    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(url_for('index'))

    pdf_files = []
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_batch')

    if os.path.exists(upload_path):
        shutil.rmtree(upload_path)
    os.makedirs(upload_path)

    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_path, filename)
                file.save(filepath)

                if filename.endswith('.zip'):
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(upload_path)
                    os.remove(filepath)

                    for root, dirs, files_in_zip in os.walk(upload_path):
                        for f in files_in_zip:
                            if f.endswith('.pdf'):
                                pdf_files.append(os.path.join(root, f))
                else:
                    pdf_files.append(filepath)

        if not pdf_files:
            flash('No PDF files found')
            return redirect(url_for('index'))

        results = process_pdfs(pdf_files)

        shutil.rmtree(upload_path)

        return render_template('results.html', results=results)

    except Exception as e:
        flash(f'Error processing files: {str(e)}')
        return redirect(url_for('index'))

def process_pdfs(pdf_files):
    pdf_handler = PDFHandler()
    text_processor = TextProcessor()

    results = {
        'files': [],
        'similarity_matrix': None,
        'average_ai_score': 0
    }

    texts = []

    for pdf_path in pdf_files:
        try:
            text = pdf_handler.extract_text(pdf_path)
            texts.append(text)

            processed_text = text_processor.preprocess_for_model(text)

            if model and tokenizer:
                ai_score = get_ai_score(processed_text)
            else:
                ai_score = 0.0

            results['files'].append({
                'name': os.path.basename(pdf_path),
                'ai_score': ai_score,
                'text_preview': text[:200] + '...' if len(text) > 200 else text
            })

        except Exception as e:
            results['files'].append({
                'name': os.path.basename(pdf_path),
                'ai_score': 0.0,
                'error': str(e)
            })

    # Calculate similarity matrix if more than one file
    if len(texts) > 1:
        results['similarity_matrix'] = calculate_similarity_matrix(texts)

    # Calculate average AI score
    valid_scores = [f['ai_score'] for f in results['files'] if 'error' not in f]
    results['average_ai_score'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    return results

def get_ai_score(text):
    """Get AI-generated content probability using the Hugging Face model"""
    try:
        # Truncate text if it's too long (RoBERTa has a max length of 512 tokens)
        max_length = 512
        
        # Tokenize and prepare input
        inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1)
            
            # Get the probability for AI-generated text (usually the second class)
            ai_prob = probabilities[0][1].item()
            
            # Convert to percentage
            return float(ai_prob * 100)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0.0

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets this env var
    app.run(host='0.0.0.0', port=port)
