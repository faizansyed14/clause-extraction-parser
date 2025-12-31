# Legal Contract Clause Parser

A state-of-the-art legal contract clause extraction system that accurately parses any type of legal contract (PDF/images), extracts clauses with proper numbering and hierarchies, and presents them in a clean web interface.

## Features

- **Advanced OCR**: Uses PaddleOCR-VL for layout-aware document parsing
- **Document Understanding**: LayoutLMv3 for structure analysis
- **Legal-Specific NLP**: Legal-BERT for clause extraction
- **Clause Classification**: 41 CUAD categories supported
- **Hierarchical Structure**: Maintains clause numbering and hierarchy (1, 1.1, 1.1.1)
- **Web Interface**: Clean React frontend with TypeScript
- **Docker Support**: Easy deployment with Docker Compose

## Technology Stack

### Backend
- Python 3.11+ with FastAPI
- PaddleOCR for OCR
- LayoutLMv3 for document structure
- Legal-BERT for clause extraction
- Docker for containerization

### Frontend
- React 18+ with TypeScript
- TailwindCSS for styling
- Axios for API communication

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   └── services/
│   │       ├── document_processor.py
│   │       └── clause_classifier.py
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ContractUploader.tsx
│   │   │   └── ClauseTree.tsx
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed
- (Optional) NVIDIA GPU with CUDA for faster processing

### Quick Start

1. **Clone or navigate to the project directory**

2. **Start the services using Docker Compose:**
   
   **Windows:**
   ```cmd
   start.bat
   ```
   
   **Linux/Mac:**
   ```bash
   ./start.sh
   ```
   
   Or manually:
   ```bash
   docker-compose up --build
   ```

   This will:
   - Build the backend container with all dependencies
   - Build the frontend container
   - Start both services
   - Download ML models on first run (~1GB total)

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

**Note:** First startup may take 5-10 minutes as it downloads ML models. Subsequent starts will be much faster.

### Manual Setup (Development)

#### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
cd frontend
npm install
npm start
```

## API Endpoints

### `POST /api/parse-contract`
Upload a PDF contract and receive structured clause extraction.

**Request:**
- Content-Type: `multipart/form-data`
- Body: PDF file

**Response:**
```json
{
  "clauses": [
    {
      "number": "1.1",
      "title": "Definitions",
      "content": "...",
      "type": "definitions",
      "confidence": 0.95,
      "level": 2,
      "children": []
    }
  ],
  "metadata": {
    "total_pages": 10,
    "processing_date": "2025-01-XX",
    "filename": "contract.pdf"
  },
  "total_pages": 10,
  "confidence_score": 0.92,
  "processing_time": 12.3
}
```

### `GET /api/clause-types`
Get list of all supported clause types (41 CUAD categories).

### `GET /api/health`
Health check endpoint.

## Supported Clause Types (CUAD Categories)

The system supports 41 clause categories from the CUAD dataset:

- Document Name, Parties, Agreement Date
- Effective Date, Expiration Date, Renewal Term
- Notice Period, Governing Law, Jurisdiction
- Non-Compete, Exclusivity, No-Solicit
- IP Ownership, License Grant, Liability Cap
- Indemnity, Confidentiality, Termination
- And 25+ more categories...

## Performance

- **Clause Extraction Accuracy**: Target >94%
- **Processing Speed**: <30 seconds per 10-page contract (with GPU)
- **Layout Preservation**: >90% accuracy

## Model Downloads

On first run, the system will automatically download:
- LayoutLMv3-base (~500MB)
- Legal-BERT-base-uncased (~440MB)
- PaddleOCR models (~100MB)

These are cached for subsequent runs.

## GPU Support

For GPU acceleration (optional but recommended for faster processing):

**Linux:**
- NVIDIA GPU with CUDA support
- nvidia-docker2 installed
- Uncomment GPU section in `docker-compose.yml`

**Windows:**
- Requires WSL2 with NVIDIA GPU support
- Or use CPU mode (slower but works on all systems)

The system works without GPU but processing will be slower (2-3x).

## Troubleshooting

### Models not downloading
- Check internet connection
- Ensure sufficient disk space (~2GB for all models)
- Check HuggingFace access (models are public)

### OCR errors
- Ensure PDF is not corrupted
- Try converting PDF to images manually
- Check PaddleOCR installation

### Memory issues
- Reduce batch size in processing
- Use CPU-only mode (slower but uses less memory)
- Process documents one at a time

## Development

### Adding New Clause Types

Edit `backend/app/services/clause_classifier.py` to add new categories and keywords.

### Improving OCR Accuracy

Fine-tune PaddleOCR settings in `document_processor.py`:
- Adjust DPI for image conversion
- Modify OCR parameters
- Add custom preprocessing

### Training Custom Models

1. Use CUAD dataset from HuggingFace
2. Fine-tune Legal-BERT on your specific contract types
3. Update model paths in `document_processor.py`

## License

This project is for educational and research purposes.

## References

- **CUAD Dataset**: https://huggingface.co/datasets/Hendrycks/CUAD
- **Legal-BERT**: https://huggingface.co/nlpaueb/legal-bert-base-uncased
- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
- **LayoutLMv3**: https://huggingface.co/microsoft/layoutlmv3-base

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

