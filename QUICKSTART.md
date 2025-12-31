# Quick Start Guide

## Windows Users

1. **Install Docker Desktop for Windows**
   - Download from: https://www.docker.com/products/docker-desktop
   - Make sure WSL2 backend is enabled

2. **Open PowerShell or Command Prompt in the project directory**

3. **Run:**
   ```cmd
   start.bat
   ```
   
   Or manually:
   ```cmd
   docker-compose up --build
   ```

4. **Wait for services to start** (first time: 5-10 minutes for model downloads)

5. **Open browser:** http://localhost:3000

## Linux/Mac Users

1. **Install Docker and Docker Compose**

2. **Open terminal in the project directory**

3. **Run:**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```
   
   Or manually:
   ```bash
   docker-compose up --build
   ```

4. **Wait for services to start** (first time: 5-10 minutes for model downloads)

5. **Open browser:** http://localhost:3000

## Testing the System

1. **Prepare a PDF contract** (any legal contract will work)

2. **Upload via the web interface:**
   - Click "Select PDF File"
   - Choose your contract PDF
   - Click "Parse Contract"
   - Wait for processing (typically 10-30 seconds per page)

3. **View results:**
   - Browse extracted clauses in the tree view
   - Expand clauses to see full content
   - Check clause types and confidence scores

## Troubleshooting

### Models not downloading
- Check internet connection
- Ensure Docker has internet access
- Check Docker logs: `docker-compose logs backend`

### Port already in use
- Change ports in `docker-compose.yml`:
  - Backend: `8000:8000` → `8001:8000`
  - Frontend: `3000:3000` → `3001:3000`

### Out of memory
- Close other applications
- Process smaller documents
- Use CPU mode (remove GPU config)

### Slow processing
- Normal for first run (model loading)
- CPU mode is slower than GPU
- Large documents take longer

## Next Steps

- Read the full README.md for advanced configuration
- Check API documentation at http://localhost:8000/docs
- Customize clause classification in `backend/app/services/clause_classifier.py`

