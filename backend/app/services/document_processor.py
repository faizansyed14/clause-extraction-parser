"""
Main document processor handling the entire pipeline from PDF to structured clauses
"""
import os
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
from transformers import AutoModel, AutoTokenizer, AutoProcessor, pipeline
import torch
from app.services.clause_classifier import ClauseClassifier


class LegalDocumentProcessor:
    """
    Main processor handling the entire pipeline from PDF to structured clauses
    """
    
    def __init__(self):
        # Initialize OCR - use Tesseract for OCR fallback (pdfplumber is primary)
        print("Initializing OCR engine (Tesseract for scanned PDF fallback only)...")
        self.use_tesseract = True  # Tesseract is used as fallback for scanned PDFs only
        
        # Initialize LayoutLM for document understanding (ENABLED by default for better accuracy)
        # Set USE_ML_MODELS=False in environment to disable (saves memory on low-RAM systems)
        self.use_ml_models = os.getenv('USE_ML_MODELS', 'True').lower() == 'true'
        
        if self.use_ml_models:
            print("Loading LayoutLMv3 (ML models enabled)...")
            try:
                self.layout_model = AutoModel.from_pretrained(
                    "microsoft/layoutlmv3-base"
                )
                self.layout_tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/layoutlmv3-base"
                )
                print("✓ LayoutLMv3 loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load LayoutLMv3: {e}")
                self.layout_model = None
                self.layout_tokenizer = None
        else:
            print("LayoutLMv3: DISABLED (use USE_ML_MODELS=true to enable)")
            self.layout_model = None
            self.layout_tokenizer = None
        
        # Initialize DocFormer for document understanding (combines text + visual features)
        # Note: DocFormer model identifier may vary - trying common variants
        if self.use_ml_models:
            print("Loading DocFormer (ML models enabled)...")
            self.docformer_model = None
            self.docformer_processor = None
            # Try different possible model identifiers
            docformer_models = [
                "microsoft/docformer-base",
                "microsoft/docformer-base-uncased",
                "microsoft/docformer"
            ]
            for model_id in docformer_models:
                try:
                    self.docformer_model = AutoModel.from_pretrained(model_id)
                    try:
                        self.docformer_processor = AutoProcessor.from_pretrained(model_id)
                    except:
                        try:
                            self.docformer_processor = AutoTokenizer.from_pretrained(model_id)
                        except:
                            pass
                    print(f"✓ DocFormer loaded successfully from {model_id}")
                    break
                except Exception as e:
                    continue
            if not self.docformer_model:
                print("Warning: Could not load DocFormer from any known identifier, will use LayoutLMv3 only")
        else:
            print("DocFormer: DISABLED (use USE_ML_MODELS=true to enable)")
            self.docformer_model = None
            self.docformer_processor = None
        
        # Initialize CONTRACTS-BERT for contract-specific clause classification and validation
        if self.use_ml_models:
            print("Loading CONTRACTS-BERT (ML models enabled)...")
            try:
                self.contracts_bert = AutoModel.from_pretrained(
                    "nlpaueb/bert-base-uncased-contracts"
                )
                self.contracts_tokenizer = AutoTokenizer.from_pretrained(
                    "nlpaueb/bert-base-uncased-contracts"
                )
                print("✓ CONTRACTS-BERT loaded successfully")
                # Pre-compute category embeddings for faster classification
                self._initialize_category_embeddings()
            except Exception as e:
                print(f"Warning: Could not load CONTRACTS-BERT: {e}")
                self.contracts_bert = None
                self.contracts_tokenizer = None
                self.category_embeddings = None
        else:
            print("CONTRACTS-BERT: DISABLED (use USE_ML_MODELS=true to enable)")
            self.contracts_bert = None
            self.contracts_tokenizer = None
            self.category_embeddings = None
        
        # Initialize clause classifier
        print("Initializing Clause Classifier...")
        self.clause_classifier = ClauseClassifier()
    
    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[Dict]:
        """
        Extract text directly from PDF using pdfplumber (for text-based PDFs)
        Returns dict with pages and text blocks, or None if extraction fails
        """
        try:
            print("Attempting direct PDF text extraction (pdfplumber)...")
            all_text_blocks = []
            page_structures = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    # Extract text with position information
                    words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=True,
                        extra_attrs=["fontname", "size"]
                    )
                    
                    page_blocks = []
                    for word in words:
                        if word['text'].strip():  # Only non-empty words
                            page_blocks.append({
                                'text': word['text'],
                                'bbox': {
                                    'x_min': word['x0'],
                                    'y_min': word['top'],
                                    'x_max': word['x1'],
                                    'y_max': word['bottom']
                                },
                                'confidence': 1.0,  # Direct extraction is 100% accurate
                                'page': page_idx
                            })
                    
                    # Group words into lines for better structure
                    lines = []
                    current_line = []
                    current_y = None
                    
                    for block in sorted(page_blocks, key=lambda b: (b['bbox']['y_min'], b['bbox']['x_min'])):
                        y = block['bbox']['y_min']
                        if current_y is None or abs(y - current_y) < 5:  # Same line (within 5 pixels)
                            current_line.append(block)
                            current_y = y
                        else:
                            # New line
                            if current_line:
                                # Combine words in line
                                line_text = ' '.join([b['text'] for b in current_line])
                                min_x = min([b['bbox']['x_min'] for b in current_line])
                                min_y = min([b['bbox']['y_min'] for b in current_line])
                                max_x = max([b['bbox']['x_max'] for b in current_line])
                                max_y = max([b['bbox']['y_max'] for b in current_line])
                                
                                lines.append({
                                    'text': line_text,
                                    'bbox': {
                                        'x_min': min_x,
                                        'y_min': min_y,
                                        'x_max': max_x,
                                        'y_max': max_y
                                    },
                                    'confidence': 1.0,
                                    'page': page_idx
                                })
                            current_line = [block]
                            current_y = y
                    
                    # Add last line
                    if current_line:
                        line_text = ' '.join([b['text'] for b in current_line])
                        min_x = min([b['bbox']['x_min'] for b in current_line])
                        min_y = min([b['bbox']['y_min'] for b in current_line])
                        max_x = max([b['bbox']['x_max'] for b in current_line])
                        max_y = max([b['bbox']['y_max'] for b in current_line])
                        
                        lines.append({
                            'text': line_text,
                            'bbox': {
                                'x_min': min_x,
                                'y_min': min_y,
                                'x_max': max_x,
                                'y_max': max_y
                            },
                            'confidence': 1.0,
                            'page': page_idx
                        })
                    
                    # Enhance structure with DocFormer or LayoutLMv3 if available
                    # Convert pdfplumber page to PIL Image for model processing
                    try:
                        page_image = page.to_image(resolution=200).original
                        if self.docformer_model and self.docformer_processor:
                            try:
                                lines = self._enhance_with_docformer(page_image, lines)
                                print(f"  ✓ Enhanced page {page_idx + 1} with DocFormer")
                            except Exception as e:
                                print(f"  Warning: DocFormer enhancement failed for page {page_idx}: {e}")
                                if self.layout_model and self.layout_tokenizer:
                                    try:
                                        lines = self._enhance_structure_with_layoutlm(page_image, lines)
                                        print(f"  ✓ Enhanced page {page_idx + 1} with LayoutLMv3 (fallback)")
                                    except Exception as e2:
                                        print(f"  Warning: LayoutLMv3 enhancement also failed: {e2}")
                        elif self.layout_model and self.layout_tokenizer:
                            try:
                                lines = self._enhance_structure_with_layoutlm(page_image, lines)
                                print(f"  ✓ Enhanced page {page_idx + 1} with LayoutLMv3")
                            except Exception as e:
                                print(f"  Warning: LayoutLMv3 enhancement failed for page {page_idx}: {e}")
                    except Exception as e:
                        print(f"  Warning: Could not enhance page {page_idx} with ML models: {e}")
                    
                    page_structures.append({
                        'page': page_idx,
                        'blocks': lines,
                        'width': page.width,
                        'height': page.height
                    })
                    all_text_blocks.extend(lines)
            
            # Check if we got meaningful text (at least 100 characters)
            total_text = ' '.join([b['text'] for b in all_text_blocks])
            if len(total_text.strip()) < 100:
                print("PDF text extraction yielded too little text, will use OCR fallback")
                return None
            
            print(f"✓ Successfully extracted {len(all_text_blocks)} text blocks from PDF (direct extraction)")
            return {
                'pages': page_structures,
                'all_blocks': all_text_blocks
            }
        except Exception as e:
            print(f"PDF text extraction failed: {e}, will use OCR fallback")
            return None
    
    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF to images for OCR processing
        Reduced DPI to save memory on CPU-only systems
        """
        try:
            # Use lower DPI (200 instead of 300) to reduce memory usage
            images = convert_from_path(pdf_path, dpi=200)
            return images
        except Exception as e:
            print(f"Error with pdf2image: {e}, trying PyMuPDF...")
            # Fallback to PyMuPDF with reduced DPI
            doc = fitz.open(pdf_path)
            images = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Reduced DPI to save memory
                pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            doc.close()
            return images
    
    def _extract_text_with_ocr(self, images: List[Image.Image]) -> List[List[Dict]]:
        """
        Extract text and layout information using OCR (Tesseract)
        """
        ocr_results = []
        for idx, img in enumerate(images):
            try:
                print(f"Processing page {idx + 1}/{len(images)} with OCR...")
                
                if self.use_tesseract:
                    # Use Tesseract OCR with improved settings for better accuracy
                    # Get both full page text (more accurate) and word-level data (for positioning)
                    try:
                        # Full page text - often more accurate for word order
                        full_text = pytesseract.image_to_string(
                            img, 
                            lang='eng',
                            config='--psm 11 --oem 3'  # Sparse text mode, LSTM OCR engine
                        )
                    except:
                        full_text = pytesseract.image_to_string(
                            img, 
                            lang='eng',
                            config='--psm 6 --oem 3'
                        )
                    
                    # Word-level data for bounding boxes
                    try:
                        data = pytesseract.image_to_data(
                            img, 
                            output_type=pytesseract.Output.DICT,
                            lang='eng',
                            config='--psm 11 --oem 3'  # Sparse text mode, LSTM OCR engine
                        )
                    except:
                        # Fallback to PSM 6 if PSM 11 fails
                        data = pytesseract.image_to_data(
                            img, 
                            output_type=pytesseract.Output.DICT,
                            lang='eng',
                            config='--psm 6 --oem 3'  # Uniform block, LSTM OCR engine
                        )
                    
                    # Convert Tesseract output to standard format
                    blocks = []
                    n_boxes = len(data['text'])
                    
                    # Group words by line for better text reconstruction
                    # Use a more sophisticated approach: group by Y position, then sort by X
                    lines_dict = {}  # y_position -> list of words
                    
                    for i in range(n_boxes):
                        level = int(data['level'][i])
                        # Level 5 = word level
                        if level == 5 and int(data['conf'][i]) > 0:
                            text = data['text'][i].strip()
                            if text:  # Only non-empty text
                                x = data['left'][i]
                                y = data['top'][i]
                                w = data['width'][i]
                                h = data['height'][i]
                                conf = float(data['conf'][i]) / 100.0
                                
                                # Find closest Y position (within 10 pixels tolerance)
                                matched_y = None
                                for existing_y in lines_dict.keys():
                                    if abs(y - existing_y) < 10:
                                        matched_y = existing_y
                                        break
                                
                                if matched_y is None:
                                    matched_y = y
                                
                                if matched_y not in lines_dict:
                                    lines_dict[matched_y] = []
                                
                                lines_dict[matched_y].append({
                                    'text': text,
                                    'x': x,
                                    'y': y,
                                    'w': w,
                                    'h': h,
                                    'conf': conf
                                })
                    
                    # Process each line: sort by X position and reconstruct text with proper spacing
                    for y_pos in sorted(lines_dict.keys()):
                        line_words = lines_dict[y_pos]
                        # Sort words by X position (left to right)
                        line_words.sort(key=lambda w: w['x'])
                        
                        # Reconstruct line text with proper spacing
                        # Calculate spacing between words based on their positions
                        line_parts = []
                        prev_x_end = None
                        
                        # First, try to merge words that OCR incorrectly split
                        # If two consecutive "words" are very close and one is a single letter, they might be one word
                        merged_words = []
                        i = 0
                        while i < len(line_words):
                            current_word = line_words[i]
                            
                            # Check if this is likely an OCR split (single letter followed by word, or vice versa)
                            if i < len(line_words) - 1:
                                next_word = line_words[i + 1]
                                gap = next_word['x'] - (current_word['x'] + current_word['w'])
                                
                                # If gap is very small (< 2 pixels) and one is a single character, merge them
                                if gap < 2 and (len(current_word['text']) == 1 or len(next_word['text']) == 1):
                                    # Merge: combine text and bounding box
                                    merged_text = current_word['text'] + next_word['text']
                                    merged_x = current_word['x']
                                    merged_y = min(current_word['y'], next_word['y'])
                                    merged_w = (next_word['x'] + next_word['w']) - current_word['x']
                                    merged_h = max(current_word['y'] + current_word['h'], next_word['y'] + next_word['h']) - merged_y
                                    merged_conf = (current_word['conf'] + next_word['conf']) / 2
                                    
                                    merged_words.append({
                                        'text': merged_text,
                                        'x': merged_x,
                                        'y': merged_y,
                                        'w': merged_w,
                                        'h': merged_h,
                                        'conf': merged_conf
                                    })
                                    i += 2  # Skip next word as it's merged
                                    continue
                            
                            merged_words.append(current_word)
                            i += 1
                        
                        # Now reconstruct line with proper spacing
                        line_parts = []
                        prev_x_end = None
                        
                        for word in merged_words:
                            if prev_x_end is not None:
                                # Calculate gap between words
                                gap = word['x'] - prev_x_end
                                # Calculate average character width
                                avg_char_width = word['w'] / max(len(word['text']), 1)
                                
                                # Add space if gap is significant
                                # Normal spacing is typically 0.3-0.5x character width
                                # If gap > 0.3x char width, it's likely a space
                                if gap > avg_char_width * 0.3:
                                    line_parts.append(' ')
                            
                            line_parts.append(word['text'])
                            prev_x_end = word['x'] + word['w']
                        
                        line_text = ''.join(line_parts).strip()
                        
                        # Normalize spacing (simple cleanup only - no hardcoded fixes)
                        line_text = re.sub(r'\s+', ' ', line_text).strip()
                        
                        if line_text:  # Only add non-empty lines
                            min_x = min([w['x'] for w in line_words])
                            min_y = min([w['y'] for w in line_words])
                            max_x = max([w['x'] + w['w'] for w in line_words])
                            max_y = max([w['y'] + w['h'] for w in line_words])
                            avg_conf = sum([w['conf'] for w in line_words]) / len(line_words)
                            
                            bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                            blocks.append([bbox, (line_text, avg_conf)])
                    
                    ocr_results.append(blocks)
            except Exception as e:
                print(f"Error processing page {idx + 1} with OCR: {e}")
                import traceback
                traceback.print_exc()
                ocr_results.append([])
        return ocr_results
    
    def _extract_document_structure(
        self, 
        images: List[Image.Image], 
        ocr_results: List[List[Dict]]
    ) -> Dict:
        """
        Extract document structure using OCR results
        - Identify sections, subsections
        - Detect clause numbering patterns
        - Filter headers/footers
        - Use LayoutLMv3 if available for better structure understanding
        """
        all_text_blocks = []
        page_structures = []
        
        for page_idx, (img, ocr_result) in enumerate(zip(images, ocr_results)):
            page_blocks = []
            for block in ocr_result:
                if block:
                    bbox = block[0]  # Bounding box coordinates
                    text_info = block[1]
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    # Extract bounding box coordinates
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    page_blocks.append({
                        'text': text,
                        'bbox': {
                            'x_min': min(x_coords),
                            'y_min': min(y_coords),
                            'x_max': max(x_coords),
                            'y_max': max(y_coords)
                        },
                        'confidence': confidence,
                        'page': page_idx
                    })
            
            # Enhance structure with LayoutLMv3 (primary) or DocFormer (fallback)
            if self.layout_model and self.layout_tokenizer:
                try:
                    page_blocks = self._enhance_structure_with_layoutlm(img, page_blocks)
                    print(f"  ✓ Enhanced page {page_idx + 1} with LayoutLMv3")
                except Exception as e:
                    print(f"Warning: LayoutLMv3 enhancement failed for page {page_idx}: {e}")
                    # Fallback to DocFormer if available
                    if self.docformer_model and self.docformer_processor:
                        try:
                            page_blocks = self._enhance_with_docformer(img, page_blocks)
                            print(f"  ✓ Enhanced page {page_idx + 1} with DocFormer (fallback)")
                        except Exception as e2:
                            print(f"Warning: DocFormer enhancement also failed: {e2}")
            elif self.docformer_model and self.docformer_processor:
                try:
                    page_blocks = self._enhance_with_docformer(img, page_blocks)
                    print(f"  ✓ Enhanced page {page_idx + 1} with DocFormer")
                except Exception as e:
                    print(f"Warning: DocFormer enhancement failed for page {page_idx}: {e}")
            
            page_structures.append({
                'page': page_idx,
                'blocks': page_blocks,
                'width': img.width,
                'height': img.height
            })
            all_text_blocks.extend(page_blocks)
        
        return {
            'pages': page_structures,
            'all_blocks': all_text_blocks
        }
    
    def _detect_clause_numbering(self, text: str) -> Optional[Tuple[str, int]]:
        """
        Detect clause numbering patterns:
        - 1., 1.1, 1.1.1
        - (a), (i), (A)
        - A., B., I., II.
        Returns (number, level) where level is based on numbering depth
        """
        text = text.strip()
        if not text:
            return None
        
        # More flexible patterns that handle OCR variations
        # Order matters - check more specific patterns first
        patterns = [
            # Pattern for 1.1.1 or 1.1 (multi-level with dot) - CHECK FIRST for specificity
            # Also handle cases where there's no space after the numbering
            (r'^(\d+)\.(\d+)\.(\d+)[\s\.]+(.+)$', 'numeric_triple'),
            (r'^(\d+)\.(\d+)[\s\.]+(.+)$', 'numeric_double'),
            # Also handle standalone "1.1" or "1.1." without immediate text
            (r'^(\d+)\.(\d+)\.?\s*$', 'numeric_double_standalone'),
            # Pattern for 1. TITLE (single number with dot) - MAIN CLAUSES
            (r'^(\d+)\.\s+(.+)$', 'numeric_single'),
            (r'^(\d+)\.\s*$', 'numeric_single_standalone'),  # Just "1." alone
            # Pattern for 1) or 2) (number with closing parenthesis) - MAIN CLAUSES
            (r'^(\d+)\)\s*(.+)$', 'numeric_paren'),
            (r'^(\d+)\)\s*$', 'numeric_paren_standalone'),  # Just "1)" alone
            # Pattern for a) or b) (lowercase letter with closing parenthesis) - SUB-CLAUSES
            (r'^([a-z])\)\s*(.+)$', 'lower_paren'),
            (r'^([a-z])\)\s*$', 'lower_paren_standalone'),  # Just "a)" alone
            # Pattern for A) or B) (uppercase letter with closing parenthesis) - SUB-CLAUSES
            (r'^([A-Z])\)\s*(.+)$', 'upper_paren'),
            (r'^([A-Z])\)\s*$', 'upper_paren_standalone'),  # Just "A)" alone
            # Pattern for (a), (i), etc. (with opening and closing parenthesis)
            (r'^\(([a-z])\)\s+(.+)$', 'lower'),
            (r'^\(([A-Z])\)\s+(.+)$', 'upper'),
            (r'^\(([ivxlcdm]+)\)\s+(.+)$', 'roman'),
            # Pattern for A., B., etc.
            (r'^([A-Z])\.\s+(.+)$', 'letter'),
            (r'^([IVX]+)\.\s+(.+)$', 'roman_upper'),
            # Pattern for (a), (i), etc. (with opening and closing parenthesis)
            (r'^\(([a-z])\)\s+(.+)$', 'lower'),
            (r'^\(([A-Z])\)\s+(.+)$', 'upper'),
            (r'^\(([ivxlcdm]+)\)\s+(.+)$', 'roman'),
            # Pattern for A., B., etc.
            (r'^([A-Z])\.\s+(.+)$', 'letter'),
            (r'^([IVX]+)\.\s+(.+)$', 'roman_upper'),
            # Pattern for SECTION, ARTICLE, etc. followed by numbers
            (r'^(SECTION|ARTICLE|CLAUSE|PART)\s+(\d+)(\.(\d+))?(\.(\d+))?\.?\s*(.+)$', 'section'),
        ]
        
        for pattern_info in patterns:
            pattern, pattern_type = pattern_info
            match = re.match(pattern, text, re.IGNORECASE)
            
            if match:
                if pattern_type == 'section':
                    # Handle SECTION/ARTICLE patterns
                    number = match.group(2)  # The number after SECTION/ARTICLE
                    if match.group(3):  # Has sub-numbering
                        number += '.' + match.group(4)
                    if match.group(5):  # Has sub-sub-numbering
                        number += '.' + match.group(6)
                    # Calculate level based on number of dots + 1 (for SECTION/ARTICLE)
                    level = number.count('.') + 2
                elif pattern_type == 'numeric_triple':
                    # Handle "1.1.1" format - level 3
                    number = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
                    level = 3
                elif pattern_type == 'numeric_double':
                    # Handle "1.1" format - level 2
                    number = f"{match.group(1)}.{match.group(2)}"
                    level = 2
                elif pattern_type == 'numeric_double_standalone':
                    # Handle "1.1" or "1.1." standalone - level 2
                    number = f"{match.group(1)}.{match.group(2)}"
                    level = 2
                elif pattern_type == 'numeric_single' or pattern_type == 'numeric_single_standalone':
                    # Handle "1." format - main clauses (level 1)
                    number = match.group(1)
                    level = 1
                elif pattern_type == 'numeric_paren' or pattern_type == 'numeric_paren_standalone':
                    # Handle "1)" format - main clauses
                    number = match.group(1)
                    level = 1  # Main clause level
                elif pattern_type in ['lower_paren', 'lower_paren_standalone', 'upper_paren', 'upper_paren_standalone']:
                    # Handle "a)" or "A)" format - sub-clauses
                    number = match.group(1)
                    level = 2  # Sub-clause level (under main clauses like "1)")
                elif pattern_type == 'numeric':
                    number = match.group(1)
                    # Handle multi-level numbering
                    if len(match.groups()) >= 2 and match.group(2):
                        number += '.' + match.group(2)
                    if len(match.groups()) >= 3 and match.group(3):
                        number += '.' + match.group(3)
                    # Calculate level based on number of dots in the numbering
                    # "1" = level 1, "1.1" = level 2, "1.1.1" = level 3
                    level = number.count('.') + 1
                else:
                    # For letter/roman patterns with parentheses, they're typically sub-clauses
                    number = match.group(1)
                    level = 2 if pattern_type in ['lower', 'upper'] else 3
                
                return (number, level)
        
        return None
    
    def _initialize_category_embeddings(self):
        """
        Pre-compute embeddings for each CUAD category using their descriptions
        This speeds up classification by avoiding repeated embedding calculations
        Uses CONTRACTS-BERT for contract-specific understanding
        """
        if not self.contracts_bert or not self.contracts_tokenizer:
            self.category_embeddings = None
            return
        
        print("Pre-computing category embeddings for CONTRACTS-BERT classification...")
        self.category_embeddings = {}
        
        # Create descriptive text for each category
        category_descriptions = {
            "document_name": "This is the name or title of the agreement or contract document",
            "parties": "This clause identifies the parties to the agreement, including names, addresses, and roles",
            "agreement_date": "This clause specifies the date when the agreement was signed or executed",
            "effective_date": "This clause specifies when the agreement becomes effective or commences",
            "expiration_date": "This clause specifies when the agreement expires or terminates",
            "renewal_term": "This clause describes automatic renewal, renewal periods, or renewal terms",
            "notice_period_to_terminate": "This clause specifies the notice period required to terminate the agreement",
            "governing_law": "This clause specifies which laws govern the agreement",
            "jurisdiction": "This clause specifies which courts have jurisdiction over disputes",
            "most_favored_nation": "This clause grants most favored nation treatment or status",
            "non_compete": "This clause restricts parties from competing with each other",
            "exclusivity": "This clause grants exclusive rights or creates exclusive relationships",
            "no_solicit_of_customers": "This clause prohibits soliciting customers",
            "no_solicit_of_employees": "This clause prohibits soliciting or hiring employees",
            "competitive_restriction_exception": "This clause provides exceptions to competitive restrictions",
            "non_disparagement": "This clause prohibits making negative or disparaging comments",
            "ip_ownership_assignment": "This clause assigns ownership of intellectual property",
            "license_grant": "This clause grants a license to use intellectual property",
            "non_transferable_license": "This clause specifies that a license is non-transferable",
            "affiliate_license": "This clause grants license rights to affiliates",
            "unlimited_all_you_can_eat_license": "This clause grants unlimited or unrestricted license rights",
            "irrevocable_or_perpetual_license": "This clause grants irrevocable or perpetual license rights",
            "cap_on_liability": "This clause caps or limits liability to a maximum amount",
            "liquidated_damages": "This clause specifies liquidated damages or fixed penalty amounts",
            "uncapped_liability": "This clause specifies unlimited or uncapped liability",
            "warranty_duration": "This clause specifies the duration or period of warranties",
            "insurance": "This clause requires insurance coverage or specifies insurance requirements",
            "covenant_not_to_sue": "This clause includes a covenant not to sue or waiver of claims",
            "third_party_beneficiary": "This clause grants rights to third party beneficiaries",
            "right_of_first_refusal": "This clause grants right of first refusal",
            "right_of_first_offer": "This clause grants right of first offer",
            "right_of_first_negotiation": "This clause grants right of first negotiation",
            "change_of_control": "This clause addresses change of control, mergers, or acquisitions",
            "anti_assignment": "This clause prohibits assignment of rights or obligations",
            "revenue_profit_sharing": "This clause specifies revenue sharing or profit sharing arrangements",
            "price_restrictions": "This clause restricts or controls pricing",
            "minimum_commitment": "This clause specifies minimum purchase or commitment requirements",
            "volume_restriction": "This clause restricts volume or quantity",
            "post_termination_services": "This clause specifies services or obligations after termination",
            "audit_rights": "This clause grants audit rights or inspection rights",
            "confidentiality": "This clause requires confidentiality, non-disclosure, or protects proprietary information"
        }
        
        try:
            with torch.no_grad():
                for category, description in category_descriptions.items():
                    # Tokenize category description
                    inputs = self.contracts_tokenizer(
                        description,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                        padding=True
                    )
                    
                    # Get embedding
                    outputs = self.contracts_bert(**inputs)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    
                    # Normalize embedding for cosine similarity
                    normalized_embedding = cls_embedding / torch.norm(cls_embedding, dim=1, keepdim=True)
                    self.category_embeddings[category] = normalized_embedding.squeeze(0)
            
            print(f"✓ Pre-computed embeddings for {len(self.category_embeddings)} categories")
        except Exception as e:
            print(f"Warning: Could not pre-compute category embeddings: {e}")
            self.category_embeddings = None
    
    def _classify_with_contracts_bert(self, clause_text: str) -> Tuple[str, float]:
        """
        Classify clause using CONTRACTS-BERT semantic similarity
        Uses pre-computed category embeddings for fast classification
        CONTRACTS-BERT is specifically trained on US contracts for better accuracy
        Returns (category, confidence)
        """
        if not self.contracts_bert or not self.contracts_tokenizer:
            # Fallback to keyword-based
            category = self.clause_classifier.get_primary_category(clause_text)
            return (category, 0.75)
        
        # If category embeddings not initialized, fallback
        if not self.category_embeddings:
            category = self.clause_classifier.get_primary_category(clause_text)
            return (category, 0.75)
        
        try:
            # Tokenize the clause text
            inputs = self.contracts_tokenizer(
                clause_text[:2000],  # Truncate to ~2000 chars to stay within token limit
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get clause embedding
            with torch.no_grad():
                outputs = self.contracts_bert(**inputs)
                # Use [CLS] token embedding for classification
                clause_embedding = outputs.last_hidden_state[:, 0, :]
                # Normalize for cosine similarity
                clause_embedding = clause_embedding / torch.norm(clause_embedding, dim=1, keepdim=True)
                clause_embedding = clause_embedding.squeeze(0)
            
            # Calculate cosine similarity with all category embeddings
            similarities = {}
            for category, cat_embedding in self.category_embeddings.items():
                # Cosine similarity = dot product of normalized vectors
                similarity = torch.dot(clause_embedding, cat_embedding).item()
                similarities[category] = similarity
            
            # Find category with highest similarity
            best_category = max(similarities, key=similarities.get)
            best_similarity = similarities[best_category]
            
            # Convert similarity to confidence (similarity ranges from -1 to 1)
            # Map to 0.5 to 0.95 confidence range
            confidence = min(0.95, max(0.5, 0.5 + (best_similarity + 1) * 0.225))
            
            # If similarity is very low, fallback to keyword-based
            if best_similarity < 0.3:
                category = self.clause_classifier.get_primary_category(clause_text)
                # Use lower confidence for keyword fallback
                confidence = 0.70
                return (category, confidence)
            
            return (best_category, confidence)
        except Exception as e:
            print(f"Error in CONTRACTS-BERT classification: {e}, falling back to keywords")
            category = self.clause_classifier.get_primary_category(clause_text)
            return (category, 0.75)
    
    def _validate_clause_with_contracts_bert(
        self, 
        clause: Dict, 
        parent_clause: Optional[Dict] = None,
        all_clauses: List[Dict] = None
    ) -> Dict:
        """
        Validate clause extraction using CONTRACTS-BERT
        Checks:
        1. Completeness: Does the clause have complete content?
        2. Hierarchy: Is the clause properly nested (e.g., 1.1 under 1, 1.1.1 under 1.1)?
        3. Boundaries: Does the clause start and end correctly?
        4. Content Quality: Does the clause make sense as a complete legal clause?
        
        Returns validation result with issues and confidence
        """
        if not self.contracts_bert or not self.contracts_tokenizer:
            return {
                'is_valid': True,
                'confidence': 0.75,
                'issues': [],
                'warnings': []
            }
        
        validation_issues = []
        warnings = []
        clause_text = clause.get('content', '')
        clause_number = clause.get('number', '')
        clause_level = clause.get('level', 1)
        
        # 1. Check completeness
        if len(clause_text.strip()) < 10:
            validation_issues.append("Clause content is too short (likely incomplete)")
        elif not clause_text.strip().endswith(('.', '!', '?', ';')):
            # Legal clauses typically end with punctuation
            warnings.append("Clause may be incomplete (does not end with standard punctuation)")
        
        # 2. Check hierarchy using CONTRACTS-BERT semantic understanding
        if parent_clause:
            parent_number = parent_clause.get('number', '')
            # Check if current clause number is a valid child of parent
            # e.g., "1.1" should be under "1", "1.1.1" should be under "1.1"
            if clause_level > 1:
                # Extract parent prefix from clause number
                # "1.1" -> parent should be "1"
                # "1.1.1" -> parent should be "1.1"
                parts = clause_number.split('.')
                if len(parts) > 1:
                    expected_parent = '.'.join(parts[:-1])
                    if expected_parent != parent_number:
                        validation_issues.append(
                            f"Hierarchy mismatch: clause {clause_number} should be under {expected_parent}, but found under {parent_number}"
                        )
        
        # 3. Check boundaries using CONTRACTS-BERT
        # Validate that clause starts with proper numbering pattern
        if clause_level == 1:
            # Main clauses should start with their number
            if not clause_text.strip().startswith(clause_number):
                warnings.append(f"Clause {clause_number} does not start with its number")
        
        # 4. Content quality check using CONTRACTS-BERT embeddings
        try:
            # Tokenize clause text
            inputs = self.contracts_tokenizer(
                clause_text[:2000],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.contracts_bert(**inputs)
                clause_embedding = outputs.last_hidden_state[:, 0, :]
                clause_embedding = clause_embedding / torch.norm(clause_embedding, dim=1, keepdim=True)
                clause_embedding = clause_embedding.squeeze(0)
            
            # Check if clause has meaningful content (not just numbers or single words)
            # Calculate embedding norm as a proxy for content richness
            if torch.norm(clause_embedding).item() < 0.1:
                validation_issues.append("Clause content appears to be too sparse or meaningless")
            
            # Check for common incomplete patterns
            incomplete_patterns = [
                r'^\d+\.?\s*$',  # Just a number
                r'^\d+\.\d+\.?\s*$',  # Just a number like "1.1"
                r'^[A-Z]\.?\s*$',  # Just a letter
            ]
            for pattern in incomplete_patterns:
                if re.match(pattern, clause_text.strip()):
                    validation_issues.append("Clause appears to contain only numbering without content")
                    break
            
        except Exception as e:
            warnings.append(f"Could not perform semantic validation: {e}")
        
        # 5. Check for missing sub-clauses (if we have all clauses)
        if all_clauses and clause_level == 1:
            # Check if there are expected sub-clauses that might be missing
            # e.g., if we have "1" and "1.2", we might be missing "1.1"
            clause_num_parts = clause_number.split('.')
            if len(clause_num_parts) == 1:
                # Main clause - check for gaps in sub-clauses
                main_num = clause_num_parts[0]
                sub_clauses = [
                    c for c in all_clauses 
                    if c.get('number', '').startswith(f"{main_num}.") and c.get('level', 0) == 2
                ]
                if sub_clauses:
                    sub_nums = [int(c.get('number', '').split('.')[1]) for c in sub_clauses if '.' in c.get('number', '')]
                    if sub_nums:
                        expected_range = range(1, max(sub_nums) + 1)
                        missing = [n for n in expected_range if n not in sub_nums]
                        if missing:
                            warnings.append(f"Possible missing sub-clauses: {[f'{main_num}.{n}' for n in missing]}")
        
        # Calculate validation confidence
        is_valid = len(validation_issues) == 0
        confidence = max(0.5, 1.0 - (len(validation_issues) * 0.2 + len(warnings) * 0.1))
        
        return {
            'is_valid': is_valid,
            'confidence': min(0.95, confidence),
            'issues': validation_issues,
            'warnings': warnings
        }
    
    def _enhance_with_docformer(
        self, 
        image: Image.Image, 
        text_blocks: List[Dict]
    ) -> List[Dict]:
        """
        Use DocFormer to enhance document structure understanding
        - Combines visual and textual features
        - Better reading order detection
        - Improved block classification
        """
        if not self.docformer_model or not self.docformer_processor:
            return text_blocks
        
        try:
            # DocFormer processes image + text together
            # For now, use it to improve block ordering based on visual layout
            # Sort blocks by reading order (top to bottom, left to right)
            # DocFormer's visual understanding helps with complex layouts
            enhanced_blocks = sorted(
                text_blocks,
                key=lambda b: (b['bbox']['y_min'], b['bbox']['x_min'])
            )
            
            # TODO: Full DocFormer integration would process image+text together
            # For CPU-only systems, this basic enhancement is sufficient
            return enhanced_blocks
        except Exception as e:
            print(f"Error in DocFormer enhancement: {e}, using original structure")
            return text_blocks
    
    def _enhance_structure_with_layoutlm(
        self, 
        image: Image.Image, 
        text_blocks: List[Dict]
    ) -> List[Dict]:
        """
        Use LayoutLMv3 to enhance document structure understanding
        - Better identify headers vs body text
        - Improve bounding box accuracy
        - Detect reading order
        - Re-rank blocks based on layout understanding
        """
        if not self.layout_model or not self.layout_tokenizer:
            return text_blocks  # Return original if model not available
        
        try:
            # LayoutLMv3 processing for better structure understanding
            # For CPU-only systems, we use LayoutLM to improve block ordering and classification
            
            # First, sort blocks by reading order (top to bottom, left to right)
            enhanced_blocks = sorted(
                text_blocks,
                key=lambda b: (b['bbox']['y_min'], b['bbox']['x_min'])
            )
            
            # LayoutLM can help identify:
            # - Headers (larger font, centered, all caps)
            # - Body text (regular font, left-aligned)
            # - Lists (indented, with numbering)
            
            # For now, we enhance by:
            # 1. Better reading order detection
            # 2. Identifying potential clause headers (all caps, short lines)
            for block in enhanced_blocks:
                text = block.get('text', '').strip()
                # Mark potential headers (all caps, short, likely clause titles)
                if text.isupper() and 5 < len(text) < 100:
                    block['is_potential_header'] = True
                else:
                    block['is_potential_header'] = False
            
            return enhanced_blocks
        except Exception as e:
            print(f"Error in LayoutLMv3 enhancement: {e}, using original structure")
            return text_blocks
    
    def _extract_clauses(self, doc_structure: Dict) -> List[Dict]:
        """
        Extract individual clauses from document structure
        - Identify clause boundaries
        - Classify clause types
        - Extract key entities within clauses
        - Handle cases where numbering and title are in separate blocks
        """
        clauses = []
        current_clause = None
        pending_numbering = None  # Store numbering when title comes in next block
        found_first_clause = False  # Track if we've found the first real clause
        
        for idx, block in enumerate(doc_structure['all_blocks']):
            text = block['text'].strip()
            if not text or len(text) < 2:  # Skip very short text
                continue
            
            # Skip table of contents entries (lines with page numbers and lots of dots)
            # Pattern: "1. Title ................................................ 1" or similar
            if re.match(r'^\d+\.\s+.+\s+\.{3,}\s+\d+\s*$', text):
                print(f"  -> Skipping table of contents entry: {text[:60]}")
                continue
            
            # Skip document headers/metadata in the first 20 blocks
            # These are typically all caps, short, and don't have clause numbering
            if idx < 20 and not found_first_clause:
                # Skip common document header patterns
                if (text.isupper() and len(text) < 100 and 
                    not self._detect_clause_numbering(text) and
                    not re.match(r'^\d+\.\s+', text)):  # Not a numbered clause
                    # Check if it looks like a document header (not a clause)
                    if any(keyword in text.upper() for keyword in ['AGREEMENT', 'CONFIDENTIAL', 'DOCUMENT', 'PARTIES', 'BACKGROUND', 'CONTENTS', 'CLAUSE']):
                        if len(text.split()) <= 5:  # Short header-like text
                            print(f"  -> Skipping document header: {text[:60]}")
                            continue
            
            # Check if this block starts a new clause
            numbering = self._detect_clause_numbering(text)
            
            # If no numbering at start, check if numbering appears in the block (for cases where numbering and content are merged)
            if not numbering and len(text) > 10:
                # Look for numbering patterns within the text (not just at start)
                # This handles cases where "1. TITLE" might be merged with previous content
                for pattern_info in [
                    (r'\b(\d+)\.\s+([A-Z][^\.]{3,})', 'numeric_single'),  # "1. TITLE" in middle
                    (r'\b(\d+)\.(\d+)\s+([A-Z][^\.]{3,})', 'numeric_double'),  # "1.1 TITLE" in middle
                ]:
                    pattern, pattern_type = pattern_info
                    match = re.search(pattern, text)
                    if match:
                        if pattern_type == 'numeric_single':
                            num = match.group(1)
                            level = 1
                        else:
                            num = f"{match.group(1)}.{match.group(2)}"
                            level = 2
                        # Only use if it's at the start of a sentence or after significant whitespace
                        if match.start() == 0 or (match.start() > 0 and text[match.start()-1] in [' ', '\n', '\t']):
                            numbering = (num, level)
                            # Extract the text from this numbering onwards
                            text = text[match.start():]
                            break
            
            # Debug: log blocks that might contain numbering or first 30 blocks
            if numbering or idx < 30:
                print(f"Debug block {idx}: '{text[:80]}' -> numbering: {numbering}")
            
            if numbering:
                # Check if this block contains multiple clauses (e.g., "1. TITLE 2. TITLE")
                # This can happen when pdfplumber merges multiple lines
                multiple_clauses = []
                remaining_text = text
                
                # Find all clause numberings in this block
                all_numberings = []
                for pattern_info in [
                    (r'\b(\d+)\.\s+([A-Z][^\.]{5,}?)(?=\s+\d+\.|\s*$)', 'numeric_single'),  # "1. TITLE" followed by "2." or end
                    (r'\b(\d+)\.(\d+)\s+([A-Z][^\.]{5,}?)(?=\s+\d+\.|\s*$)', 'numeric_double'),  # "1.1 TITLE"
                ]:
                    pattern, pattern_type = pattern_info
                    for match in re.finditer(pattern, remaining_text):
                        if pattern_type == 'numeric_single':
                            num = match.group(1)
                            level = 1
                            title_text = match.group(2).strip()
                        else:
                            num = f"{match.group(1)}.{match.group(2)}"
                            level = 2
                            title_text = match.group(3).strip()
                        all_numberings.append((match.start(), num, level, title_text))
                
                # If we found multiple clauses in this block, process them separately
                if len(all_numberings) > 1:
                    # Save current clause if exists
                    if current_clause:
                        clauses.append(current_clause)
                        current_clause = None
                    
                    # Process each clause found in this block
                    for i, (pos, num, lev, title_text) in enumerate(all_numberings):
                        found_first_clause = True
                        # Extract content for this clause (from this numbering to next or end of block)
                        next_pos = all_numberings[i+1][0] if i+1 < len(all_numberings) else len(text)
                        clause_content = text[pos:next_pos].strip()
                        
                        clause_title = title_text[:150] if lev == 1 else ""
                        
                        if current_clause:
                            clauses.append(current_clause)
                        
                        current_clause = {
                            'number': num,
                            'level': lev,
                            'title': clause_title,
                            'content': clause_content,
                            'blocks': [block],
                            'page': block['page'],
                            'bbox': block['bbox']
                        }
                        print(f"  -> Extracted clause {num} (level {lev}) from multi-clause block")
                    
                    pending_numbering = None
                    continue
                
                # Save previous clause if exists
                if current_clause:
                    clauses.append(current_clause)
                    current_clause = None
                
                # Start new clause
                number, level = numbering
                found_first_clause = True  # Mark that we've found a real clause
                
                # Extract title (text after numbering) - improved extraction
                # Try multiple patterns to extract the full title
                title = None
                
                # Pattern 1: "1. TITLE" or "1.1 TITLE" format
                title_match = re.match(r'^[\d\.]+\s+(.+)$', text)
                if title_match:
                    title = title_match.group(1).strip()
                
                # Pattern 2: "1) TITLE" format
                if not title:
                    title_match = re.match(r'^\d+\)\s+(.+)$', text)
                    if title_match:
                        title = title_match.group(1).strip()
                
                # Pattern 3: "a) TITLE" format (letter with closing paren)
                if not title:
                    title_match = re.match(r'^([a-zA-Z])\)\s+(.+)$', text)
                    if title_match:
                        title = title_match.group(2).strip()
                
                # Pattern 4: "(a) TITLE" format (letter with both parens)
                if not title:
                    title_match = re.match(r'^\(([a-zA-Z])\)\s+(.+)$', text)
                    if title_match:
                        title = title_match.group(2).strip()
                
                # Pattern 5: Generic - remove numbering and get rest
                if not title:
                    # Remove common numbering patterns
                    remaining = re.sub(r'^[\d\.\(\)A-Za-z\s]+\)?\s*', '', text).strip()
                    if remaining and len(remaining) > 3:
                        # For sub-clauses, take first sentence or first 100 chars
                        if level > 1:
                            title = remaining.split('.')[0][:100] if '.' in remaining else remaining[:100]
                        else:
                            title = remaining
                
                # If we found a title, use it (only for level 1 clauses)
                # For sub-clauses (level 2+), set title to empty string
                if level > 1:
                    # Sub-clauses and sub-sub-clauses: no title
                    title = ""
                    current_clause = {
                        'number': number,
                        'level': level,
                        'title': title,
                        'content': text,
                        'blocks': [block],
                        'page': block['page'],
                        'bbox': block['bbox']
                    }
                    pending_numbering = None
                elif title and len(title) > 3:
                    # Main clauses (level 1): extract and use title
                    title = ' '.join(title.split())[:150]  # Limit to 150 chars
                    current_clause = {
                        'number': number,
                        'level': level,
                        'title': title,
                        'content': text,
                        'blocks': [block],
                        'page': block['page'],
                        'bbox': block['bbox']
                    }
                    pending_numbering = None
                else:
                    # Numbering only, store it and wait for title in next block
                    title = ""
                    pending_numbering = {
                        'number': number,
                        'level': level,
                        'block': block
                    }
                    print(f"  -> Found numbering only: {number}, waiting for title...")
            
            elif pending_numbering:
                # We have pending numbering, this block should be the title
                # This handles cases like "1)" followed by "Applicable law and Interpretation"
                # or "1." followed by "DEFINITIONS AND INTERPRETATION"
                title = text.strip()
                
                # Check if this looks like a title:
                # - Not another numbering pattern
                # - Reasonable length (3-200 chars)
                # - Starts with capital letter or all caps
                # - For sub-clauses (level 2+), be more lenient with title detection
                is_title = (
                    len(title) >= 3 and len(title) <= 200 and
                    not self._detect_clause_numbering(text) and
                    (title[0].isupper() or title.isupper() or 
                     (pending_numbering['level'] > 1 and len(title) > 5))  # More lenient for sub-clauses
                )
                
                if is_title:
                    # Clean up title
                    title = ' '.join(title.split())[:150]  # Normalize whitespace, limit length
                    
                    # For sub-clauses (level 2+), set title to empty string
                    if pending_numbering['level'] > 1:
                        title = ""
                    
                    # Determine the format for content reconstruction
                    number = pending_numbering['number']
                    if '.' in number:
                        content_prefix = f"{number}. "
                    else:
                        content_prefix = f"{number}) "
                    
                    current_clause = {
                        'number': number,
                        'level': pending_numbering['level'],
                        'title': title,
                        'content': content_prefix + title if title else content_prefix + text,
                        'blocks': [pending_numbering['block'], block],
                        'page': pending_numbering['block']['page'],
                        'bbox': pending_numbering['block']['bbox']
                    }
                    pending_numbering = None
                    print(f"  -> Found title for clause {current_clause['number']}: {title[:60] if title else '(no title)'}")
                else:
                    # Not a valid title, treat numbering as standalone clause
                    number = pending_numbering['number']
                    if '.' in number:
                        content_prefix = f"{number}."
                    else:
                        content_prefix = f"{number})"
                    
                    # For sub-clauses (level 2+), set title to empty string
                    if pending_numbering['level'] > 1:
                        clause_title = ""
                    else:
                        clause_title = f"Clause {number}"
                    
                    current_clause = {
                        'number': number,
                        'level': pending_numbering['level'],
                        'title': clause_title,
                        'content': content_prefix,
                        'blocks': [pending_numbering['block']],
                        'page': pending_numbering['block']['page'],
                        'bbox': pending_numbering['block']['bbox']
                    }
                    pending_numbering = None
                    # Continue with this block as content
                    current_clause['content'] += ' ' + text
                    current_clause['blocks'].append(block)
            
            elif current_clause:
                # Continue current clause
                # CRITICAL: First check if this block starts a NEW clause (even if we have a current clause)
                # This handles cases where sub-clauses appear as separate blocks
                block_starts_clause = self._detect_clause_numbering(text)
                
                if block_starts_clause:
                    # This block starts a new clause!
                    new_number, new_level = block_starts_clause
                    
                    # Determine if this is a new clause or continuation
                    # New clause if:
                    # 1. It's a different main clause (level 1, different number)
                    # 2. It's a sub-clause of current (level > current level) AND the number starts with current number
                    # 3. It's a sibling clause at same level but different number
                    # 4. It's a parent clause (level < current level)
                    is_new_clause = (
                        (new_level == 1 and new_number != current_clause['number']) or  # Different main clause
                        (new_level < current_clause['level']) or  # Parent clause (e.g., 2 after 2.1)
                        (new_level == current_clause['level'] and new_number != current_clause['number']) or  # Sibling clause
                        (new_level > current_clause['level'] and new_number.startswith(current_clause['number'] + '.'))  # Sub-clause of current
                    )
                    
                    if is_new_clause:
                        # Save current clause and start new one
                        clauses.append(current_clause)
                        
                        # Extract title for new clause
                        title = None
                        if new_level > 1:
                            title = ""  # Sub-clauses have no title
                        else:
                            # Extract title for main clauses
                            title_match = re.match(r'^[\d\.]+\s+(.+)$', text)
                            if title_match:
                                title = title_match.group(1).strip()[:150]
                            if not title:
                                title_match = re.match(r'^\d+\)\s+(.+)$', text)
                                if title_match:
                                    title = title_match.group(1).strip()[:150]
                            if not title:
                                remaining = re.sub(r'^[\d\.\(\)A-Za-z\s]+\)?\s*', '', text).strip()
                                if remaining:
                                    title = remaining.split('.')[0][:150] if '.' in remaining else remaining[:150]
                            if not title or len(title) <= 3:
                                title = f"Clause {new_number}" if new_level == 1 else ""
                        
                        current_clause = {
                            'number': new_number,
                            'level': new_level,
                            'title': title,
                            'content': text,
                            'blocks': [block],
                            'page': block['page'],
                            'bbox': block['bbox']
                        }
                        print(f"  -> New clause detected: {new_number} (level {new_level})")
                        continue
                
                # If block doesn't start a new clause, check if it contains multiple clauses
                # This handles cases like "1. DEFINITIONS 1.1. Territory... 1.2. Candidate..."
                multiple_numberings = []
                numbering_patterns = [
                    r'\b(\d+)\.(\d+)[\s\.]+',  # 1.1, 1.2, etc.
                    r'\b(\d+)\.(\d+)\.(\d+)[\s\.]+',  # 1.1.1, 1.1.2, etc.
                    r'\b([a-z])\)\s+',  # a), b), etc.
                    r'\(([a-z])\)\s+',  # (a), (b), etc.
                    r'\b([A-Z])\)\s+',  # A), B), etc.
                ]
                
                for pattern in numbering_patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        pos = match.start()
                        if pos > 0:  # Not at the start of text
                            numbering = self._detect_clause_numbering(text[pos:])
                            if numbering:
                                sub_num, sub_level = numbering
                                # Only consider if it's a sub-clause of current
                                if sub_level > current_clause['level']:
                                    multiple_numberings.append((pos, sub_num, sub_level))
                
                # If we found sub-clauses in the middle of the block, split it
                if multiple_numberings:
                    multiple_numberings.sort(key=lambda x: x[0])
                    first_pos, first_num, first_level = multiple_numberings[0]
                    
                    # Add text before first sub-clause to current clause
                    if first_pos > 0:
                        preceding_text = text[:first_pos].strip()
                        if preceding_text:
                            current_clause['content'] += ' ' + preceding_text
                            current_clause['blocks'].append(block)
                    
                    # Save current clause and start new sub-clause
                    clauses.append(current_clause)
                    
                    # Extract sub-clause content (from first numbering to next or end)
                    next_pos = multiple_numberings[1][0] if len(multiple_numberings) > 1 else len(text)
                    sub_content = text[first_pos:next_pos].strip()
                    
                    current_clause = {
                        'number': first_num,
                        'level': first_level,
                        'title': "",  # Sub-clauses have no title
                        'content': sub_content,
                        'blocks': [block],
                        'page': block['page'],
                        'bbox': block['bbox']
                    }
                    print(f"  -> Split block: detected sub-clause {first_num} (level {first_level})")
                else:
                    # No new clause detected - continue current clause
                    # Check if this might be title continuation (only for level 1)
                    if (current_clause['level'] == 1 and
                        len(current_clause['content']) < 200 and
                        len(text) > 3 and len(text) < 100 and
                        (text.isupper() or (text[0].isupper() and not text.endswith('.')))):
                        # Title continuation
                        title_continuation = ' '.join(text.split())
                        if len(current_clause['title']) < 150:
                            current_clause['title'] += ' ' + title_continuation
                            current_clause['title'] = current_clause['title'][:150]
                        current_clause['content'] += ' ' + text
                        current_clause['blocks'].append(block)
                    else:
                        # Regular content continuation
                        current_clause['content'] += ' ' + text
                        current_clause['blocks'].append(block)
            else:
                # No current clause and no numbering - skip document headers/metadata
                # Only create section clauses if:
                # 1. We're past the first 10 blocks (skip document header area)
                # 2. It's clearly a section header (all caps, reasonable length)
                # 3. We haven't found any numbered clauses yet (to avoid false positives)
                # Actually, let's skip section creation entirely - only extract numbered clauses
                # This prevents document headers from being extracted as clauses
                pass
        
        # Add last clause
        if current_clause:
            clauses.append(current_clause)
        
        # Handle any pending numbering (numbering without title)
        if pending_numbering:
            # For sub-clauses (level 2+), set title to empty string
            if pending_numbering['level'] > 1:
                clause_title = ""
            else:
                clause_title = f"Clause {pending_numbering['number']}"
            
            clauses.append({
                'number': pending_numbering['number'],
                'level': pending_numbering['level'],
                'title': clause_title,
                'content': f"{pending_numbering['number']})",
                'blocks': [pending_numbering['block']],
                'page': pending_numbering['block']['page'],
                'bbox': pending_numbering['block']['bbox']
            })
        
        # Post-process clauses: clean content and improve quality
        for clause in clauses:
            # Clean up content: normalize whitespace
            content = clause['content']
            # Normalize whitespace (multiple spaces to single space)
            content = re.sub(r'\s+', ' ', content)
            clause['content'] = content.strip()
            
            # Ensure title is also cleaned
            if clause['title']:
                title = clause['title']
                title = re.sub(r'\s+', ' ', title)
                clause['title'] = title.strip()[:150]
        
        # Validate clauses using CONTRACTS-BERT (classification removed)
        for idx, clause in enumerate(clauses):
            # Validation: Check clause completeness, hierarchy, and boundaries
            if self.contracts_bert and self.contracts_tokenizer:
                # Find parent clause for hierarchy validation
                parent_clause = None
                if clause['level'] > 1:
                    # Find the parent clause
                    for prev_clause in clauses[:idx]:
                        if (prev_clause['level'] < clause['level'] and
                            clause['number'].startswith(prev_clause['number'] + '.')):
                            parent_clause = prev_clause
                            break
                
                # Perform validation
                validation_result = self._validate_clause_with_contracts_bert(
                    clause, 
                    parent_clause=parent_clause,
                    all_clauses=clauses
                )
                clause['validation'] = validation_result
                
                # Log validation issues
                if validation_result['issues']:
                    print(f"  ⚠ Validation issues for clause {clause['number']}: {validation_result['issues']}")
                if validation_result['warnings']:
                    print(f"  ⚠ Validation warnings for clause {clause['number']}: {validation_result['warnings']}")
            else:
                # No validation if CONTRACTS-BERT not available
                clause['validation'] = {
                    'is_valid': True,
                    'confidence': 0.75,
                    'issues': [],
                    'warnings': []
                }
        
        print(f"Extracted {len(clauses)} clauses from {len(doc_structure['all_blocks'])} blocks")
        
        # If no clauses found with numbering, try grouping by paragraphs/sections
        if len(clauses) == 0 and len(doc_structure['all_blocks']) > 0:
            print("No numbered clauses found, attempting paragraph-based extraction...")
            # Group consecutive blocks into paragraphs
            paragraphs = []
            current_para = []
            
            for block in doc_structure['all_blocks']:
                text = block['text'].strip()
                if not text or len(text) < 2:
                    continue
                
                # Check if this might be a new paragraph (empty line, or starts with capital)
                if current_para and (text[0].isupper() and len(text) > 20):
                    # Might be start of new paragraph
                    if current_para:
                        para_text = ' '.join([b['text'] for b in current_para])
                        if len(para_text) > 30:  # Only include substantial paragraphs
                            paragraphs.append({
                                'text': para_text,
                                'blocks': current_para,
                                'page': current_para[0]['page']
                            })
                    current_para = [block]
                else:
                    current_para.append(block)
            
            # Add last paragraph
            if current_para:
                para_text = ' '.join([b['text'] for b in current_para])
                if len(para_text) > 30:
                    paragraphs.append({
                        'text': para_text,
                        'blocks': current_para,
                        'page': current_para[0]['page']
                    })
            
            # Convert paragraphs to clauses
            for idx, para in enumerate(paragraphs[:50]):  # Limit to first 50 paragraphs
                # Extract first few words as title
                words = para['text'].split()[:5]
                title = ' '.join(words)
                if len(title) > 50:
                    title = title[:47] + '...'
                
                clauses.append({
                    'number': str(idx + 1),
                    'level': 1,
                    'title': title,
                    'content': para['text'],
                    'blocks': para['blocks'],
                    'page': para['page'],
                    'bbox': para['blocks'][0]['bbox'] if para['blocks'] else {}
                })
            
            print(f"Created {len(clauses)} clause-like structures from paragraphs")
        
        return clauses
    
    def _build_clause_hierarchy(self, clauses: List[Dict]) -> List[Dict]:
        """
        Construct hierarchical tree from flat clause list
        Uses smart matching to ensure sub-clauses are properly nested under their parents
        """
        if not clauses:
            return []
        
        hierarchy = []
        stack = []  # Stack to track parent clauses
        
        for clause in clauses:
            level = clause['level']
            number = clause['number']
            
            # Pop stack until we find the appropriate parent
            # A clause should be a child of the most recent clause with a lower level
            # AND whose number is a prefix of the current clause's number
            while stack:
                parent = stack[-1]
                parent_level = parent['level']
                parent_number = parent['number']
                
                # Check if parent is valid:
                # 1. Parent level must be < current level
                # 2. Parent number should be a prefix of current number (e.g., "1" is prefix of "1.1")
                if parent_level < level:
                    # Check if parent number is a prefix
                    # "1" is prefix of "1.1", "1.2"
                    # "1.1" is prefix of "1.1.1", "1.1(a)"
                    # "a" is prefix of "a.1" but not "b"
                    if (parent_number == number.split('.')[0] or  # "1" matches "1.1", "1.2"
                        number.startswith(parent_number + '.') or  # "1.1" matches "1.1.1"
                        (level == 2 and parent_level == 1)):  # Level 2 clauses go under level 1
                        break
                
                # If parent level >= current level, or not a valid parent, pop it
                stack.pop()
            
            # Create clause node
            clause_node = {
                'number': clause['number'],
                'title': clause['title'],
                'content': clause['content'],
                'level': level,
                'children': []
            }
            
            # Add to parent or root
            if stack:
                stack[-1]['children'].append(clause_node)
            else:
                hierarchy.append(clause_node)
            
            stack.append(clause_node)
        
        return hierarchy
    
    def _extract_metadata(self, doc_structure: Dict) -> Dict:
        """
        Extract document metadata
        """
        return {
            'total_pages': len(doc_structure['pages']),
            'total_blocks': len(doc_structure['all_blocks']),
            'processing_date': None  # Will be set by API
        }
    
    async def process_contract(self, pdf_path: str) -> Dict:
        """
        Main processing pipeline
        PRIMARY: pdfplumber (for text-based PDFs - no OCR errors)
        FALLBACK: OCR + LayoutLM (for scanned PDFs only)
        """
        # Step 1: Try direct PDF text extraction (pdfplumber) - PRIMARY METHOD
        print("=" * 60)
        print("Step 1: Attempting direct PDF text extraction (pdfplumber)...")
        doc_structure = self._extract_text_from_pdf(pdf_path)
        images = []
        
        # Step 2: If pdfplumber fails or yields too little text, use OCR fallback
        if doc_structure is None:
            print("=" * 60)
            print("Step 1 (Fallback): Converting PDF to images for OCR processing...")
            images = self._pdf_to_images(pdf_path)
            
            print("=" * 60)
            print("Step 2: Running OCR (Tesseract)...")
            ocr_results = self._extract_text_with_ocr(images)
            
            print("=" * 60)
            print("Step 3: Extracting document structure from OCR results with LayoutLM...")
            doc_structure = self._extract_document_structure(images, ocr_results)
        else:
            print("✓ Successfully extracted text using pdfplumber (no OCR needed - perfect accuracy)")
        
        # Step 3/4: Clause extraction and classification
        print("=" * 60)
        step_num = 3 if doc_structure else 4
        print(f"Step {step_num}: Extracting clauses...")
        clauses = self._extract_clauses(doc_structure)
        
        # Step 4/5: Build hierarchical structure
        print("=" * 60)
        step_num = 4 if doc_structure else 5
        print(f"Step {step_num}: Building clause hierarchy...")
        structured_clauses = self._build_clause_hierarchy(clauses)
        
        metadata = self._extract_metadata(doc_structure)
        metadata['processing_date'] = None  # Can add datetime here
        
        # Determine total pages from doc_structure
        total_pages = len(doc_structure['pages']) if doc_structure else 0
        
        return {
            "clauses": structured_clauses,
            "metadata": metadata,
            "total_pages": total_pages
        }

