#!/usr/bin/env python3
"""
Challenge 1B: Persona-Driven Document Intelligence
Enhanced version with advanced features and optimizations
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from typing import List, Dict, Tuple, Any, Optional
import argparse
import pickle
import hashlib
from collections import defaultdict, Counter
import networkx as nx
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModel
import torch
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Enhanced data structure for document sections"""
    document: str
    section_title: str
    page_number: int
    context: str
    full_text: str
    relevance_score: float = 0.0
    citation_count: int = 0
    accessibility_tags: List[str] = None
    cross_references: List[str] = None
    
    def __post_init__(self):
        if self.accessibility_tags is None:
            self.accessibility_tags = []
        if self.cross_references is None:
            self.cross_references = []

class AdvancedDocumentProcessor:
    def __init__(self, cache_dir: str = "./cache", enable_multilingual: bool = False):
        """Initialize the enhanced document processor"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_multilingual = enable_multilingual
        self.concept_graph = nx.Graph()
        self.citation_network = nx.DiGraph()
        
        try:
            logger.info("Initializing advanced models...")
            self._load_models()
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def _load_models(self):
        """Load and cache models with optimization"""
        # Load spaCy with caching
        cache_key = "spacy_model"
        cached_model = self._load_from_cache(cache_key)
        
        if cached_model:
            self.nlp = cached_model
            logger.info("Loaded spaCy model from cache")
        else:
            self.nlp = spacy.load("en_core_web_sm")
            self._save_to_cache(cache_key, self.nlp)
            logger.info("Loaded and cached spaCy model")
        
        # Load sentence transformer with INT8 quantization simulation
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        

        # Load Qwen2-0.5B
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        self.qwen_model = AutoModel.from_pretrained("Qwen/Qwen2-0.5B")
        self.device = torch.device('cpu') 
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.qwen_model.to(self.device)
    
        # Create sentence_model wrapper for compatibility
        # self.sentence_model = self._create_qwen_wrapper()
        # Create a custom encode method
        def qwen_encode(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            
            with torch.no_grad():
                inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                    max_length=512, return_tensors='pt')
                outputs = self.qwen_model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.numpy()
        logger.info("Loaded sentence transformer model")
        
        # Initialize TF-IDF for keyword-based relevance
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        # Initialize topic model for persona-topic priors
        self.topic_model = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            max_iter=10  # Fast convergence for hackathon
        )
        
        # Multilingual support
        if self.enable_multilingual:
            self._setup_multilingual_support()

    def _create_qwen_wrapper(self):
        """Create a wrapper to make Qwen2 compatible with SentenceTransformer interface"""
        class QwenWrapper:
            def __init__(self, tokenizer, model, device):
                self.tokenizer = tokenizer
                self.model = model
                self.device = device
            
            def encode(self, texts, batch_size=32, show_progress_bar=False, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                
                all_embeddings = []
                
                # Process in batches to handle memory
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            batch_texts, 
                            padding=True, 
                            truncation=True, 
                            max_length=512, 
                            return_tensors='pt'
                        )
                        
                        # Move to device
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        
                        # Mean pooling
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                        
                        # Move back to CPU and convert to numpy
                        embeddings = embeddings.cpu().numpy()
                        all_embeddings.append(embeddings)
                
                # Concatenate all batches
                return np.concatenate(all_embeddings, axis=0)
        
        return QwenWrapper(self.tokenizer, self.qwen_model, self.device)
    def _setup_multilingual_support(self):
        """Setup multilingual processing capabilities"""
        logger.info("Setting up multilingual support...")
        # Language detection patterns
        self.language_patterns = {
            'japanese': re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]'),
            'chinese': re.compile(r'[\u4E00-\u9FFF]'),
            'arabic': re.compile(r'[\u0600-\u06FF]'),
            'cyrillic': re.compile(r'[\u0400-\u04FF]'),
        }
        
        # BPE tokenization simulation (simplified)
        self.bpe_vocab = self._create_bpe_vocab()

    def _create_bpe_vocab(self) -> Dict[str, int]:
        """Create simplified BPE vocabulary for multilingual support"""
        # This is a simplified version - in production, use a proper BPE tokenizer
        common_subwords = [
            'ing', 'tion', 'ness', 'ment', 'able', 'ible', 'ous', 'ful',
            'less', 'ize', 'ise', 'ly', 'er', 'est', 'ed', 'pre', 'un',
            'dis', 're', 'over', 'under', 'out', 'up', 'down', 'in', 'on'
        ]
        return {subword: i for i, subword in enumerate(common_subwords)}

    def _load_from_cache(self, key: str) -> Optional[Any]:
        """Load object from cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {key}: {e}")
        return None

    def _save_to_cache(self, key: str, obj: Any):
        """Save object to cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {key}: {e}")

    def _detect_language(self, text: str) -> str:
        """Detect language of text using Unicode block heuristics"""
        if not text:
            return 'english'
        
        for lang, pattern in self.language_patterns.items():
            if pattern.search(text):
                return lang
        
        return 'english'

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF with enhanced error handling"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        text_by_page = {}
        
        # Try multiple PDF extraction methods
        try:
            # Primary method: PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        text_by_page[page_num] = text
            
            if not text_by_page:
                raise ValueError("No text extracted with PyPDF2")
                
            logger.info(f"Extracted text from {len(text_by_page)} pages")
            
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            # Fallback: Could implement pdfplumber or other methods here
            
        return text_by_page

    def extract_text_from_file(self, file_path: str) -> Dict[int, str]:
        """Enhanced text extraction with caching and language detection"""
        cache_key = f"text_extract_{hashlib.md5(file_path.encode()).hexdigest()}"
        cached_result = self._load_from_cache(cache_key)
        
        if cached_result:
            logger.info(f"Loaded cached text extraction for {file_path}")
            return cached_result
        
        logger.info(f"Extracting text from file: {file_path}")
        
        if file_path.lower().endswith('.pdf'):
            text_by_page = self.extract_text_from_pdf(file_path)
        else:
            # Handle text files with language detection
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Detect language
                detected_lang = self._detect_language(content)
                logger.info(f"Detected language: {detected_lang}")
                
                # Split into pseudo-pages
                page_size = 1500 if detected_lang == 'english' else 1000  # Adjust for different languages
                text_by_page = {}
                for i in range(0, len(content), page_size):
                    page_num = (i // page_size) + 1
                    text_by_page[page_num] = content[i:i + page_size]
                    
                logger.info(f"Extracted text from {len(text_by_page)} pseudo-pages")
                
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                text_by_page = {}
        
        # Cache the result
        self._save_to_cache(cache_key, text_by_page)
        return text_by_page

    def detect_sections_with_accessibility(self, text: str) -> List[DocumentSection]:
        """Enhanced section detection with accessibility tagging"""
        logger.debug("Detecting sections with accessibility features...")
        sections = []
        lines = text.split('\n')
        
        # Enhanced patterns for different document types
        heading_patterns = {
            'academic': [
                r'^\d+\.?\s+[A-Z][\w\s]{5,50}$',  # Numbered headings
                r'^[IVX]+\.?\s+[A-Z][\w\s]{5,50}$',  # Roman numerals
                r'^(Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References)$',
                r'^[A-Z][A-Z\s]{10,50}$',  # ALL CAPS
            ],
            'business': [
                r'^(Executive Summary|Financial Overview|Market Analysis|Risk Factors)$',
                r'^\d{4}\s+(Q[1-4]|Quarter\s+[1-4]).*$',  # Quarterly reports
                r'^[A-Z][\w\s]{5,50}:$',  # Colon endings
            ],
            'technical': [
                r'^Chapter\s+\d+.*$',
                r'^Section\s+\d+.*$',
                r'^\d+\.\d+\s+.*$',  # Subsection numbering
            ]
        }
        
        # Detect document type
        doc_type = self._classify_document_type(text)
        patterns = heading_patterns.get(doc_type, heading_patterns['academic'])
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # Check patterns
            is_heading = False
            heading_level = 'h3'  # Default
            
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_heading = True
                    # Determine heading level for accessibility
                    if re.match(r'^[IVX]+\.?', line) or line.isupper():
                        heading_level = 'h1'
                    elif re.match(r'^\d+\.?\s+', line):
                        heading_level = 'h2'
                    break
            
            # Additional heuristics
            if not is_heading and self._is_likely_heading(line, lines, i):
                is_heading = True
                heading_level = 'h3'
            
            if is_heading:
                # Extract context
                start_idx = max(0, i - 2)
                end_idx = min(len(lines), i + 20)
                context = '\n'.join(lines[start_idx:end_idx])
                context = re.sub(r'\n+', '\n', context).strip()
                
                # Generate accessibility tags
                accessibility_tags = [heading_level]
                if self._contains_technical_content(context):
                    accessibility_tags.append('technical')
                if self._contains_numerical_data(context):
                    accessibility_tags.append('data-heavy')
                
                # Detect cross-references
                cross_refs = self._extract_cross_references(context)
                
                # Create enhanced section object
                section = DocumentSection(
                    document="",  # Will be set by caller
                    section_title=line,
                    page_number=0,  # Will be set by caller
                    context=context,
                    full_text=text,
                    accessibility_tags=accessibility_tags,
                    cross_references=cross_refs
                )
                sections.append(section)
        
        logger.debug(f"Detected {len(sections)} sections with accessibility features")
        return sections

    def _classify_document_type(self, text: str) -> str:
        """Classify document type for better section detection"""
        academic_indicators = ['abstract', 'methodology', 'references', 'citation', 'hypothesis']
        business_indicators = ['revenue', 'profit', 'market share', 'quarterly', 'financial']
        technical_indicators = ['chapter', 'section', 'algorithm', 'implementation']
        
        text_lower = text.lower()
        
        academic_score = sum(1 for indicator in academic_indicators if indicator in text_lower)
        business_score = sum(1 for indicator in business_indicators if indicator in text_lower)
        technical_score = sum(1 for indicator in technical_indicators if indicator in text_lower)
        
        if academic_score >= business_score and academic_score >= technical_score:
            return 'academic'
        elif business_score >= technical_score:
            return 'business'
        else:
            return 'technical'

    def _is_likely_heading(self, line: str, lines: List[str], index: int) -> bool:
        """Advanced heuristics for heading detection"""
        if len(line) > 100 or len(line.split()) > 12:
            return False
        
        # Check formatting context
        prev_line = lines[index - 1].strip() if index > 0 else ""
        next_line = lines[index + 1].strip() if index < len(lines) - 1 else ""
        
        # Standalone line with proper capitalization
        if (len(line.split()) >= 2 and 
            line[0].isupper() and 
            not line.endswith('.') and
            prev_line == "" and  # Empty line before
            len(next_line) > 20):  # Substantial content after
            return True
        
        return False

    def _contains_technical_content(self, text: str) -> bool:
        """Check if text contains technical content"""
        technical_terms = [
            'algorithm', 'implementation', 'framework', 'methodology',
            'analysis', 'evaluation', 'optimization', 'performance'
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in technical_terms)

    def _contains_numerical_data(self, text: str) -> bool:
        """Check if text contains significant numerical data"""
        # Count numbers, percentages, and equations
        numbers = re.findall(r'\d+\.?\d*%?', text)
        return len(numbers) > 3

    def _extract_cross_references(self, text: str) -> List[str]:
        """Extract cross-references from text"""
        # Look for section references, figure references, etc.
        patterns = [
            r'(?i)section\s+\d+(?:\.\d+)*',
            r'(?i)figure\s+\d+',
            r'(?i)table\s+\d+',
            r'(?i)chapter\s+\d+',
            r'(?i)appendix\s+[A-Z]',
        ]
        
        cross_refs = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            cross_refs.extend(matches)
        
        return list(set(cross_refs))  # Remove duplicates

    def extract_citations(self, text: str) -> List[str]:
        """Extract citations from text for citation-aware ranking"""
        citation_patterns = [
            r'\[(\d+(?:,\s*\d+)*)\]',  # [1, 2, 3]
            r'\(([A-Za-z]+\s+et\s+al\.?,?\s+\d{4})\)',  # (Smith et al., 2023)
            r'\(([A-Za-z]+,?\s+\d{4})\)',  # (Smith, 2023)
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return citations

    def build_concept_graph(self, sections: List[DocumentSection]) -> nx.Graph:
        """Build concept graph connecting related sections"""
        logger.info("Building concept graph...")
        
        # Extract key concepts from each section
        for section in sections:
            doc = self.nlp(section.context)
            
            # Extract entities and noun phrases as concepts
            concepts = []
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT', 'GPE']:
                    concepts.append(ent.text.lower().strip())
            
            # Add important noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep it manageable
                    concepts.append(chunk.text.lower().strip())
            
            # Add concepts to graph
            section_id = f"{section.document}_{section.page_number}_{section.section_title}"
            self.concept_graph.add_node(section_id, 
                                      section=section,
                                      concepts=concepts)
        
        # Connect related sections based on shared concepts
        nodes = list(self.concept_graph.nodes(data=True))
        for i, (node1, data1) in enumerate(nodes):
            for node2, data2 in nodes[i+1:]:
                # Calculate concept overlap
                concepts1 = set(data1['concepts'])
                concepts2 = set(data2['concepts'])
                overlap = len(concepts1.intersection(concepts2))
                
                if overlap > 0:
                    # Add edge with weight based on overlap
                    weight = overlap / len(concepts1.union(concepts2))
                    self.concept_graph.add_edge(node1, node2, weight=weight)
        
        logger.info(f"Built concept graph with {self.concept_graph.number_of_nodes()} nodes and {self.concept_graph.number_of_edges()} edges")
        return self.concept_graph

    def calculate_enhanced_relevance_score(self, section: DocumentSection, 
                                         persona_embedding: np.ndarray,
                                         job_embedding: np.ndarray,
                                         persona_text: str,
                                         job_text: str) -> float:
        """Enhanced relevance scoring with multiple signals"""
        try:
            logger.debug(f"Calculating enhanced relevance for: {section.section_title}")
            
            # 1. Semantic similarity (base score)
            section_text = f"{section.section_title} {section.context}"
            section_text = re.sub(r'\s+', ' ', section_text).strip()
            
            if len(section_text) < 10:
                return 0.0
            
            section_embedding = self.sentence_model.encode([section_text])
            persona_similarity = cosine_similarity(section_embedding, persona_embedding.reshape(1, -1))[0][0]
            job_similarity = cosine_similarity(section_embedding, job_embedding.reshape(1, -1))[0][0]
            
            base_score = 0.3 * persona_similarity + 0.7 * job_similarity
            
            # 2. TF-IDF keyword matching
            persona_keywords = self._extract_keywords(persona_text)
            job_keywords = self._extract_keywords(job_text)
            all_keywords = persona_keywords + job_keywords
            
            keyword_score = self._calculate_keyword_overlap(section_text, all_keywords)
            
            # 3. Citation-based importance
            citations = self.extract_citations(section.context)
            citation_score = min(len(citations) / 10, 1.0)  # Normalize to [0, 1]
            
            # 4. Cross-reference importance
            cross_ref_score = min(len(section.cross_references) / 5, 1.0)
            
            # 5. Position-based importance (earlier sections often more important)
            position_score = max(0, 1 - (section.page_number - 1) / 20)
            
            # Combine all scores with weights
            final_score = (
                0.5 * base_score +
                0.2 * keyword_score +
                0.1 * citation_score +
                0.1 * cross_ref_score +
                0.1 * position_score
            )
            
            logger.debug(f"Score breakdown - Base: {base_score:.3f}, Keyword: {keyword_score:.3f}, "
                        f"Citation: {citation_score:.3f}, Final: {final_score:.3f}")
            
            return float(final_score)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced relevance score: {e}")
            return 0.0

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        doc = self.nlp(text)
        keywords = []
        
        # Extract entities
        for ent in doc.ents:
            keywords.append(ent.text.lower())
        
        # Extract important nouns and adjectives
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ'] and 
                not token.is_stop and 
                len(token.text) > 3 and
                token.is_alpha):
                keywords.append(token.lemma_.lower())
        
        return list(set(keywords))

    def _calculate_keyword_overlap(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword overlap score"""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return min(matches / max(len(keywords), 1), 1.0)

    def extract_enhanced_subsections(self, section: DocumentSection, 
                                   persona_embedding: np.ndarray,
                                   max_subsections: int = 3) -> List[Dict]:
        """Extract subsections with persona-aware ranking"""
        logger.debug(f"Extracting enhanced subsections for: {section.section_title}")
        
        try:
            doc = self.nlp(section.context)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 30]
            
            if not sentences:
                return []
            
            # Group sentences into subsections with semantic coherence
            subsections = []
            current_subsection = []
            
            for i, sentence in enumerate(sentences):
                current_subsection.append(sentence)
                
                # End subsection at natural breaks or when reaching optimal length
                should_end = (
                    len(current_subsection) >= 3 or
                    i == len(sentences) - 1 or
                    self._is_natural_break(sentence, sentences[i+1] if i+1 < len(sentences) else "")
                )
                
                if should_end and current_subsection:
                    subsection_text = ' '.join(current_subsection).strip()
                    subsection_text = re.sub(r'\s+', ' ', subsection_text)
                    
                    if len(subsection_text) > 80:  # Substantial content
                        # Calculate relevance to persona
                        subsection_embedding = self.sentence_model.encode([subsection_text])
                        relevance = cosine_similarity(
                            subsection_embedding, 
                            persona_embedding.reshape(1, -1)
                        )[0][0]
                        
                        subsections.append({
                            'document': section.document,
                            'refined_text': subsection_text,
                            'page_number': section.page_number,
                            'relevance_score': float(relevance),
                            'accessibility_tags': section.accessibility_tags,
                            'cross_references': self._extract_cross_references(subsection_text)
                        })
                    
                    current_subsection = []
            
            # Sort by relevance and return top subsections
            subsections.sort(key=lambda x: x['relevance_score'], reverse=True)
            return subsections[:max_subsections]
            
        except Exception as e:
            logger.error(f"Error extracting enhanced subsections: {e}")
            return []

    def _is_natural_break(self, current_sentence: str, next_sentence: str) -> bool:
        """Detect natural breaks between sentences"""
        break_indicators = [
            'however', 'therefore', 'moreover', 'furthermore', 'consequently',
            'in contrast', 'on the other hand', 'additionally', 'similarly'
        ]
        
        return (
            current_sentence.endswith('.') and
            any(indicator in next_sentence.lower()[:20] for indicator in break_indicators)
        )

    def generate_explainability_report(self, selected_sections: List[DocumentSection],
                                     persona: str, job_to_be_done: str) -> Dict:
        """Generate explainability report for section selection"""
        logger.info("Generating explainability report...")
        
        explanations = []
        for i, section in enumerate(selected_sections, 1):
            explanation = {
                'rank': i,
                'section': section.section_title,
                'document': section.document,
                'relevance_score': section.relevance_score,
                'reasons': []
            }
            
            # Analyze why this section was selected
            if section.relevance_score > 0.7:
                explanation['reasons'].append("High semantic similarity to persona and job requirements")
            
            if len(section.cross_references) > 0:
                explanation['reasons'].append(f"Contains {len(section.cross_references)} cross-references indicating structural importance")
            
            if 'technical' in section.accessibility_tags:
                explanation['reasons'].append("Contains technical content relevant to the task")
            
            if 'data-heavy' in section.accessibility_tags:
                explanation['reasons'].append("Contains significant numerical/analytical data")
            
            explanations.append(explanation)
        
        return {
            'selection_criteria': {
                'persona_match': "Sections ranked by semantic similarity to persona expertise",
                'job_alignment': "Prioritized sections that directly support the job-to-be-done",
                'structural_importance': "Considered cross-references and document structure",
                'content_quality': "Filtered for substantial, relevant content"
            },
            'section_explanations': explanations
        }

    def extract_document_sections(self, file_path: str) -> List[DocumentSection]:
        """Extract sections using enhanced detection"""
        logger.info(f"Extracting document sections from: {file_path}")
        text_by_page = self.extract_text_from_file(file_path)
        all_sections = []
        
        for page_num, text in text_by_page.items():
            sections = self.detect_sections_with_accessibility(text)
            for section in sections:
                section.document = os.path.basename(file_path)
                section.page_number = page_num
                all_sections.append(section)
        
        # Build concept graph for cross-document connections
        if len(all_sections) > 0:
            self.build_concept_graph(all_sections)
        
        logger.info(f"Extracted {len(all_sections)} enhanced sections from {file_path}")
        return all_sections

    def find_document_file(self, input_dir: str, filename: str) -> str:
        """Enhanced file finding with multiple fallbacks"""
        logger.info(f"Looking for document: {filename} in {input_dir}")
        
        # Try exact match
        exact_path = os.path.join(input_dir, filename)
        if os.path.exists(exact_path):
            return exact_path
        
        # Try with .txt extension
        if filename.endswith('.pdf'):
            txt_filename = filename.replace('.pdf', '.txt')
            txt_path = os.path.join(input_dir, txt_filename)
            if os.path.exists(txt_path):
                return txt_path
        
        # Try case-insensitive search
        for file in os.listdir(input_dir):
            if file.lower() == filename.lower():
                return os.path.join(input_dir, file)
        
        logger.warning(f"Could not find document: {filename}")
        return None

    def process_documents(self, input_dir: str, input_file: str) -> Dict:
        """Enhanced main processing function with all advanced features"""
        logger.info(f"Starting enhanced document processing...")
        
        try:
            # Load configuration
            with open(input_file, 'r') as f:
                config = json.load(f)
            
            documents = config['documents']
            persona = config['persona']['role']
            job_to_be_done = config['job_to_be_done']['task']
            
            logger.info(f"Processing {len(documents)} documents")
            logger.info(f"Persona: {persona}")
            logger.info(f"Job: {job_to_be_done}")
            
            # Create enhanced embeddings
            persona_embedding = self.sentence_model.encode([persona])
            job_embedding = self.sentence_model.encode([job_to_be_done])
            
            # Extract sections from all documents
            all_sections = []
            processed_docs = []
            
            for doc_info in documents:
                file_path = self.find_document_file(input_dir, doc_info['filename'])
                if file_path:
                    sections = self.extract_document_sections(file_path)
                    all_sections.extend(sections)
                    processed_docs.append(doc_info['filename'])
                    logger.info(f"Processed: {doc_info['filename']}")
                else:
                    logger.warning(f"Document not found: {doc_info['filename']}")
            
            if not all_sections:
                raise ValueError("No content extracted from documents")
            
            # Calculate enhanced relevance scores
            logger.info("Calculating enhanced relevance scores...")
            for section in all_sections:
                section.relevance_score = self.calculate_enhanced_relevance_score(
                    section, persona_embedding[0], job_embedding[0], persona, job_to_be_done
                )
            
            # Advanced filtering and ranking
            all_sections = [s for s in all_sections if s.relevance_score > 0.15]
            
            # Implement diversified selection strategy
            top_sections = self._diversified_section_selection(all_sections, max_sections=5)
            
            logger.info(f"Selected {len(top_sections)} diverse, high-quality sections")
            
            # Extract enhanced subsections
            logger.info("Extracting enhanced subsections...")
            subsection_analysis = []
            for section in top_sections:
                subsections = self.extract_enhanced_subsections(
                    section, persona_embedding[0], max_subsections=3
                )
                subsection_analysis.extend(subsections)
            
            # Generate concept graph insights
            concept_insights = self._generate_concept_insights()
            
            # Generate explainability report
            explainability = self.generate_explainability_report(
                top_sections, persona, job_to_be_done
            )
            
            # Prepare enhanced output
            extracted_sections = []
            for i, section in enumerate(top_sections, 1):
                extracted_sections.append({
                    'document': section.document,
                    'section_title': section.section_title,
                    'importance_rank': i,
                    'page_number': section.page_number,
                    'relevance_score': round(section.relevance_score, 4),
                    'accessibility_tags': section.accessibility_tags,
                    'cross_references': section.cross_references,
                    'citation_count': len(self.extract_citations(section.context))
                })
            
            # Enhanced output with all advanced features
            output = {
                'metadata': {
                    'input_documents': processed_docs,
                    'persona': persona,
                    'job_to_be_done': job_to_be_done,
                    'processing_timestamp': datetime.now().isoformat(),
                    'model_info': {
                        'sentence_transformer': 'all-MiniLM-L6-v2',
                        'spacy_model': 'en_core_web_sm',
                        'multilingual_support': self.enable_multilingual
                    },
                    'performance_metrics': {
                        'total_sections_found': len(all_sections),
                        'sections_after_filtering': len([s for s in all_sections if s.relevance_score > 0.15]),
                        'concept_graph_nodes': self.concept_graph.number_of_nodes(),
                        'concept_graph_edges': self.concept_graph.number_of_edges()
                    }
                },
                'extracted_sections': extracted_sections,
                'subsection_analysis': subsection_analysis,
                'advanced_features': {
                    'concept_insights': concept_insights,
                    'explainability': explainability,
                    'cross_document_connections': self._get_cross_document_connections(),
                    'accessibility_summary': self._generate_accessibility_summary(top_sections)
                }
            }
            
            logger.info("Enhanced document processing complete")
            return output
            
        except Exception as e:
            logger.error(f"Error in enhanced processing: {e}")
            raise

    def _diversified_section_selection(self, sections: List[DocumentSection], 
                                     max_sections: int = 5) -> List[DocumentSection]:
        """Implement diversified selection to avoid redundancy"""
        if len(sections) <= max_sections:
            return sorted(sections, key=lambda x: x.relevance_score, reverse=True)
        
        # Step 1: Ensure representation from each document (top 2 per doc)
        sections_by_doc = defaultdict(list)
        for section in sections:
            sections_by_doc[section.document].append(section)
        
        # Sort sections within each document
        for doc_sections in sections_by_doc.values():
            doc_sections.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Select top 2 from each document
        diverse_sections = []
        for doc_sections in sections_by_doc.values():
            diverse_sections.extend(doc_sections[:2])
        
        # Step 2: Global ranking with diversity penalty
        diverse_sections.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Step 3: Apply MMR-like selection for final diversity
        selected = [diverse_sections[0]]  # Start with highest scored
        remaining = diverse_sections[1:]
        
        while len(selected) < max_sections and remaining:
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # Calculate diversity bonus (penalty for similarity to already selected)
                diversity_penalty = 0
                for selected_section in selected:
                    if candidate.document == selected_section.document:
                        diversity_penalty += 0.1  # Same document penalty
                    
                    # Semantic similarity penalty
                    candidate_embedding = self.sentence_model.encode([candidate.context])
                    selected_embedding = self.sentence_model.encode([selected_section.context])
                    similarity = cosine_similarity(candidate_embedding, selected_embedding)[0][0]
                    diversity_penalty += similarity * 0.2
                
                # Final score with diversity consideration
                final_score = candidate.relevance_score - diversity_penalty
                
                if final_score > best_score:
                    best_score = final_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected

    def _generate_concept_insights(self) -> Dict:
        """Generate insights from the concept graph"""
        if self.concept_graph.number_of_nodes() == 0:
            return {'message': 'No concept graph available'}
        
        try:
            # Find central concepts (nodes with high centrality)
            centrality = nx.betweenness_centrality(self.concept_graph)
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Find strongly connected sections
            connected_components = list(nx.connected_components(self.concept_graph))
            largest_component = max(connected_components, key=len) if connected_components else set()
            
            return {
                'central_sections': [
                    {
                        'section_id': section_id,
                        'centrality_score': round(score, 4),
                        'title': self.concept_graph.nodes[section_id]['section'].section_title
                    }
                    for section_id, score in top_central
                ],
                'connected_clusters': {
                    'largest_cluster_size': len(largest_component),
                    'total_clusters': len(connected_components),
                    'cluster_descriptions': [
                        f"Cluster {i+1}: {len(cluster)} interconnected sections"
                        for i, cluster in enumerate(connected_components[:3])
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error generating concept insights: {e}")
            return {'error': str(e)}

    def _get_cross_document_connections(self) -> List[Dict]:
        """Identify connections between different documents"""
        connections = []
        
        try:
            for edge in self.concept_graph.edges(data=True):
                node1, node2, data = edge
                section1 = self.concept_graph.nodes[node1]['section']
                section2 = self.concept_graph.nodes[node2]['section']
                
                # Only include cross-document connections
                if section1.document != section2.document:
                    connections.append({
                        'document1': section1.document,
                        'section1': section1.section_title,
                        'document2': section2.document,
                        'section2': section2.section_title,
                        'connection_strength': round(data['weight'], 4),
                        'shared_concepts': len(
                            set(self.concept_graph.nodes[node1]['concepts']).intersection(
                                set(self.concept_graph.nodes[node2]['concepts'])
                            )
                        )
                    })
            
            # Sort by connection strength
            connections.sort(key=lambda x: x['connection_strength'], reverse=True)
            return connections[:5]  # Top 5 connections
            
        except Exception as e:
            logger.error(f"Error getting cross-document connections: {e}")
            return []

    def _generate_accessibility_summary(self, sections: List[DocumentSection]) -> Dict:
        """Generate accessibility summary for selected sections"""
        try:
            tag_counts = Counter()
            total_cross_refs = 0
            
            for section in sections:
                tag_counts.update(section.accessibility_tags)
                total_cross_refs += len(section.cross_references)
            
            return {
                'content_types': dict(tag_counts),
                'accessibility_features': {
                    'heading_levels_present': [tag for tag in tag_counts if tag.startswith('h')],
                    'technical_content_sections': tag_counts.get('technical', 0),
                    'data_heavy_sections': tag_counts.get('data-heavy', 0)
                },
                'structural_elements': {
                    'total_cross_references': total_cross_refs,
                    'average_cross_refs_per_section': round(total_cross_refs / len(sections), 2) if sections else 0
                },
                'usability_notes': [
                    "Sections tagged with appropriate heading levels for screen readers",
                    "Technical content identified for specialized audience",
                    "Cross-references preserved for navigation context"
                ]
            }
        except Exception as e:
            logger.error(f"Error generating accessibility summary: {e}")
            return {'error': str(e)}

class CacheManager:
    """Advanced caching system for model optimization"""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_embedding_cache_key(self, text: str) -> str:
        """Generate cache key for text embeddings"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def cache_embeddings(self, text: str, embedding: np.ndarray):
        """Cache embeddings for reuse"""
        cache_key = self.get_embedding_cache_key(text)
        cache_file = self.cache_dir / f"emb_{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def load_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Load cached embedding if available"""
        cache_key = self.get_embedding_cache_key(text)
        cache_file = self.cache_dir / f"emb_{cache_key}.npy"
        
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        return None

def create_performance_optimized_processor(enable_caching: bool = True, 
                                         enable_multilingual: bool = False) -> AdvancedDocumentProcessor:
    """Factory function to create optimized processor"""
    cache_dir = "./cache" if enable_caching else "./cache_disabled"
    
    processor = AdvancedDocumentProcessor(
        cache_dir=cache_dir,
        enable_multilingual=enable_multilingual
    )
    
    # Add performance optimizations
    if enable_caching:
        processor.cache_manager = CacheManager()
        # Monkey patch the encode method for caching
        original_encode = processor.sentence_model.encode
        
        def cached_encode(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            
            results = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache for each text
            for i, text in enumerate(texts):
                cached = processor.cache_manager.load_cached_embedding(text)
                if cached is not None:
                    results.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Process uncached texts
            if uncached_texts:
                new_embeddings = original_encode(uncached_texts, **kwargs)
                
                # Cache new embeddings and add to results
                for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                    processor.cache_manager.cache_embeddings(text, embedding)
                    results.append((uncached_indices[idx], embedding))
            
            # Sort results by original order and extract embeddings
            results.sort(key=lambda x: x[0])
            return np.array([embedding for _, embedding in results])
        
        processor.sentence_model.encode = cached_encode
    
    return processor

def main():
    parser = argparse.ArgumentParser(description='Enhanced Persona-Driven Document Intelligence')
    parser.add_argument('--input_dir', default='/app/input', help='Input directory path')
    parser.add_argument('--output_dir', default='/app/output', help='Output directory path')
    parser.add_argument('--input_file', default='challenge1b_input.json', help='Input configuration file')
    parser.add_argument('--output_file', default='challenge1b_output.json', help='Output file name')
    parser.add_argument('--enable_caching', action='store_true', help='Enable model caching for performance')
    parser.add_argument('--enable_multilingual', action='store_true', help='Enable multilingual support')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting enhanced Challenge 1B pipeline")
        start_time = datetime.now()
        
        # Create optimized processor
        processor = create_performance_optimized_processor(
            enable_caching=args.enable_caching,
            enable_multilingual=args.enable_multilingual
        )
        
        # Process documents with enhanced features
        input_path = args.input_file
        result = processor.process_documents(args.input_dir, input_path)
        
        # Save enhanced output
        logger.info(f"Saving enhanced output to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Performance metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate performance report
        performance_report = {
            'processing_time_seconds': round(processing_time, 2),
            'documents_processed': len(result['metadata']['input_documents']),
            'sections_extracted': len(result['extracted_sections']),
            'subsections_analyzed': len(result['subsection_analysis']),
            'advanced_features_enabled': {
                'concept_graph': True,
                'accessibility_tagging': True,
                'citation_analysis': True,
                'cross_document_connections': True,
                'explainability': True,
                'multilingual': args.enable_multilingual,
                'caching': args.enable_caching
            }
        }
        
        # Save performance report
        perf_path = os.path.join(args.output_dir, 'performance_report.json')
        with open(perf_path, 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        logger.info(f"Enhanced processing complete in {processing_time:.2f}s")
        logger.info(f"Found {len(result['extracted_sections'])} relevant sections with advanced analysis")
        logger.info(f"Generated {len(result['advanced_features']['cross_document_connections'])} cross-document connections")
        
        print(f"✅ Enhanced Challenge 1B processing complete!")
        print(f"📊 Processing time: {processing_time:.2f} seconds")
        print(f"📄 Documents processed: {len(result['metadata']['input_documents'])}")
        print(f"🎯 Relevant sections found: {len(result['extracted_sections'])}")
        print(f"🔗 Cross-document connections: {len(result['advanced_features']['cross_document_connections'])}")
        print(f"🏷️ Accessibility features enabled")
        print(f"📝 Output saved to: {output_path}")
        print(f"📈 Performance report: {perf_path}")
        
    except Exception as e:
        logger.error(f"Error in enhanced main: {e}")
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()