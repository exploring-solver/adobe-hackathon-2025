#!/usr/bin/env python3
"""
Challenge 1B: Persona-Driven Document Intelligence
Enhanced version with text file support for testing
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import PyPDF2
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple, Any
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor with models"""
        try:
            logger.info("Initializing spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model")
            
            logger.info("Initializing sentence transformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF by page"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        text_by_page = {}
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    text_by_page[page_num] = text
            logger.info(f"Extracted text from {len(text_by_page)} pages in {pdf_path}")
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
        return text_by_page

    def extract_text_from_file(self, file_path: str) -> Dict[int, str]:
        """Extract text from file (PDF or text) by page"""
        logger.info(f"Extracting text from file: {file_path}")
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        else:
            # Handle text files for testing
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Split into pseudo-pages for text files (every 1000 characters)
                page_size = 1000
                text_by_page = {}
                for i in range(0, len(content), page_size):
                    page_num = (i // page_size) + 1
                    text_by_page[page_num] = content[i:i + page_size]
                logger.info(f"Extracted text from {len(text_by_page)} pseudo-pages in {file_path}")
                return text_by_page
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return {}

    def detect_sections(self, text: str) -> List[Tuple[str, int]]:
        """Detect sections/headings in text using patterns and NLP"""
        logger.debug("Detecting sections in text...")
        sections = []
        lines = text.split('\n')
        
        # Enhanced patterns for detecting headings
        heading_patterns = [
            r'^[A-Z][A-Z\s]{5,50}$',  # ALL CAPS headings
            r'^\d+\.?\s+[A-Z][\w\s]{5,50}$',  # Numbered headings
            r'^[IVX]+\.?\s+[A-Z][\w\s]{5,50}$',  # Roman numeral headings
            r'^[A-Z][\w\s]{5,50}:$',  # Headings ending with colon
            r'^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$',  # Title case headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*$'  # Multi-word title case
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 5:
                continue
                
            # Check if line matches heading patterns
            is_heading = False
            for pattern in heading_patterns:
                if re.match(pattern, line) and len(line.split()) >= 2:
                    is_heading = True
                    break
            
            # Additional heuristics for heading detection
            if not is_heading:
                # Check for standalone lines that look like headings
                if (len(line) < 100 and 
                    len(line.split()) >= 2 and 
                    len(line.split()) <= 8 and
                    line[0].isupper() and 
                    not line.endswith('.') and
                    not line.endswith(',') and
                    not any(word.lower() in line.lower() for word in ['figure', 'table', 'equation', 'page', 'chapter']) and
                    not re.match(r'^\d+$', line.strip())):  # Not just a number
                    is_heading = True
            
            if is_heading:
                sections.append((line, i))
        
        # Remove duplicates and very similar sections
        unique_sections = []
        for section_title, line_num in sections:
            is_duplicate = False
            for existing_title, _ in unique_sections:
                if (section_title.lower() == existing_title.lower() or 
                    section_title.lower() in existing_title.lower() or
                    existing_title.lower() in section_title.lower()):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_sections.append((section_title, line_num))
        
        logger.debug(f"Detected {len(sections)} raw sections, {len(unique_sections)} unique sections")
        return unique_sections

    def extract_document_sections(self, file_path: str) -> List[Dict]:
        """Extract sections from a document"""
        logger.info(f"Extracting document sections from: {file_path}")
        text_by_page = self.extract_text_from_file(file_path)
        all_sections = []
        for page_num, text in text_by_page.items():
            logger.debug(f"Processing page {page_num} of {file_path}")
            sections = self.detect_sections(text)
            for section_title, line_num in sections:
                # Extract surrounding context for the section
                lines = text.split('\n')
                start_idx = max(0, line_num - 2)
                end_idx = min(len(lines), line_num + 15)  # Increased context
                context = '\n'.join(lines[start_idx:end_idx])
                
                # Clean up context
                context = re.sub(r'\n+', '\n', context).strip()
                
                all_sections.append({
                    'document': os.path.basename(file_path),
                    'section_title': section_title,
                    'page_number': page_num,
                    'context': context,
                    'full_text': text  # Store full page text for ranking
                })
        logger.info(f"Extracted {len(all_sections)} sections from {file_path}")
        return all_sections

    def calculate_relevance_score(self, section: Dict, persona_embedding: np.ndarray, 
                                job_embedding: np.ndarray) -> float:
        """Calculate relevance score for a section based on persona and job"""
        try:
            logger.debug(f"Calculating relevance score for section: {section['section_title']}")
            # Combine section title and context for embedding
            section_text = f"{section['section_title']} {section['context']}"
            
            # Clean and prepare text for embedding
            section_text = re.sub(r'\s+', ' ', section_text).strip()
            if len(section_text) < 10:  # Too short to be meaningful
                return 0.0
                
            section_embedding = self.sentence_model.encode([section_text])
            
            # Calculate similarity with persona and job
            persona_similarity = cosine_similarity(section_embedding, persona_embedding.reshape(1, -1))[0][0]
            job_similarity = cosine_similarity(section_embedding, job_embedding.reshape(1, -1))[0][0]
            
            # Weighted combination (job is more important)
            relevance_score = 0.3 * persona_similarity + 0.7 * job_similarity
            
            logger.debug(f"Relevance score: {relevance_score:.4f}")
            return float(relevance_score)
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0

    def extract_refined_subsections(self, section: Dict, max_subsections: int = 3) -> List[Dict]:
        """Extract refined subsections from a section"""
        logger.debug(f"Extracting refined subsections for section: {section['section_title']}")
        try:
            text = section['context']
            doc = self.nlp(text)
            
            # Split into sentences and group into subsections
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
            
            if not sentences:
                return []
            
            subsections = []
            current_subsection = []
            
            for sentence in sentences:
                current_subsection.append(sentence)
                # Create subsection every 2-3 sentences or at natural breaks
                if (len(current_subsection) >= 3 or 
                    sentence.endswith('\n\n') or
                    any(marker in sentence.lower() for marker in ['however', 'therefore', 'moreover', 'furthermore'])):
                    
                    if current_subsection:
                        refined_text = ' '.join(current_subsection).strip()
                        refined_text = re.sub(r'\s+', ' ', refined_text)  # Clean whitespace
                        
                        if len(refined_text) > 50:  # Only include substantial subsections
                            subsections.append({
                                'document': section['document'],
                                'refined_text': refined_text,
                                'page_number': section['page_number']
                            })
                        current_subsection = []
            
            # Add remaining sentences as final subsection
            if current_subsection:
                refined_text = ' '.join(current_subsection).strip()
                refined_text = re.sub(r'\s+', ' ', refined_text)
                if len(refined_text) > 50:
                    subsections.append({
                        'document': section['document'],
                        'refined_text': refined_text,
                        'page_number': section['page_number']
                    })
            
            logger.debug(f"Extracted {len(subsections[:max_subsections])} refined subsections")
            return subsections[:max_subsections]
            
        except Exception as e:
            logger.error(f"Error extracting subsections: {e}")
            return []

    def find_document_file(self, input_dir: str, filename: str) -> str:
        """Find document file with flexible matching"""
        logger.info(f"Looking for document: {filename} in {input_dir}")
        # Try exact match first
        exact_path = os.path.join(input_dir, filename)
        if os.path.exists(exact_path):
            logger.info(f"Found exact match for {filename}")
            return exact_path
        
        # Try with .txt extension for testing
        if filename.endswith('.pdf'):
            txt_filename = filename.replace('.pdf', '.txt')
            txt_path = os.path.join(input_dir, txt_filename)
            if os.path.exists(txt_path):
                logger.info(f"Found .txt version for {filename}")
                return txt_path
        # Try case-insensitive search
        for file in os.listdir(input_dir):
            if file.lower() == filename.lower():
                logger.info(f"Found case-insensitive match for {filename}: {file}")
                return os.path.join(input_dir, file)
        logger.warning(f"Could not find document: {filename}")
        return None

    def process_documents(self, input_dir: str, input_file: str) -> Dict:
        """Main processing function"""
        logger.info(f"Starting document processing with input_dir={input_dir}, input_file={input_file}")
        try:
            # Load input configuration
            logger.info(f"Loading input configuration from {input_file}")
            with open(input_file, 'r') as f:
                config = json.load(f)
            documents = config['documents']
            persona = config['persona']['role']
            job_to_be_done = config['job_to_be_done']['task']
            logger.info(f"Processing {len(documents)} documents for persona: {persona}")
            logger.info(f"Job to be done: {job_to_be_done}")
            # Create embeddings for persona and job
            logger.info("Creating embeddings for persona and job description")
            persona_embedding = self.sentence_model.encode([persona])
            job_embedding = self.sentence_model.encode([job_to_be_done])
            # Extract sections from all documents
            all_sections = []
            processed_docs = []
            for doc_info in documents:
                logger.info(f"Processing document: {doc_info['filename']}")
                file_path = self.find_document_file(input_dir, doc_info['filename'])
                if file_path:
                    logger.info(f"Found file: {file_path}")
                    sections = self.extract_document_sections(file_path)
                    all_sections.extend(sections)
                    processed_docs.append(doc_info['filename'])
                    logger.info(f"Successfully processed: {doc_info['filename']}")
                else:
                    logger.warning(f"Document not found: {doc_info['filename']}")
            if not all_sections:
                logger.error("No sections extracted from any documents")
                raise ValueError("No content could be extracted from the provided documents")
            logger.info(f"Extracted {len(all_sections)} sections total from {len(processed_docs)} documents")
            # Calculate relevance scores and rank sections
            logger.info("Calculating relevance scores for all sections")
            for section in all_sections:
                section['relevance_score'] = self.calculate_relevance_score(
                    section, persona_embedding[0], job_embedding[0]
                )
            # Filter out sections with very low relevance
            logger.info("Filtering out sections with low relevance")
            all_sections = [s for s in all_sections if s['relevance_score'] > 0.1]
            # Sort by relevance score (descending)
            all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
            # Select top 5 most relevant sections Not this instead do sorting
            
            # top_sections = all_sections[:5]
            from collections import defaultdict

            # Step 1: Collect top 2 per document
            top_sections_by_doc = defaultdict(list)
            for section in all_sections:
                doc = section['document']
                if len(top_sections_by_doc[doc]) < 2:
                    top_sections_by_doc[doc].append(section)

            # Step 2: Flatten and globally sort
            flattened_sections = [sec for secs in top_sections_by_doc.values() for sec in secs]
            flattened_sections.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Step 3: Pick top 5 globally
            top_sections = flattened_sections[:5]

            logger.info(f"Selected top {len(top_sections)} sections for analysis")
            # Extract subsections for top sections
            logger.info("Extracting refined subsections for top sections")
            subsection_analysis = []
            for section in top_sections:
                subsections = self.extract_refined_subsections(section)
                subsection_analysis.extend(subsections)
            # Prepare output
            extracted_sections = []
            for i, section in enumerate(top_sections, 1):
                extracted_sections.append({
                    'document': section['document'],
                    'section_title': section['section_title'],
                    'importance_rank': i,
                    'page_number': section['page_number']
                })
            output = {
                'metadata': {
                    'input_documents': processed_docs,
                    'persona': persona,
                    'job_to_be_done': job_to_be_done,
                    'processing_timestamp': datetime.now().isoformat()
                },
                'extracted_sections': extracted_sections,
                'subsection_analysis': subsection_analysis
            }
            logger.info("Document processing complete")
            return output
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Process documents for persona-driven intelligence')
    parser.add_argument('--input_dir', default='/app/input', help='Input directory path')
    parser.add_argument('--output_dir', default='/app/output', help='Output directory path')
    parser.add_argument('--input_file', default='challenge1b_input.json', help='Input configuration file')
    parser.add_argument('--output_file', default='challenge1b_output.json', help='Output file name')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting main pipeline")
        # Initialize processor
        processor = DocumentProcessor()
        # Process documents
        input_path = args.input_file
        result = processor.process_documents(args.input_dir, input_path)
        # Save output
        logger.info(f"Saving output to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Successfully processed documents. Output saved to {output_path}")
        print(f"Processing complete. Found {len(result['extracted_sections'])} relevant sections.")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()