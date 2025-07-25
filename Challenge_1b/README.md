# Challenge 1B: Persona-Driven Document Intelligence

## Overview

This solution implements an intelligent document analysis system that extracts and prioritizes the most relevant sections from a collection of PDF documents based on a specific persona and their job-to-be-done.

## Approach

### 1. Document Processing Pipeline

**Text Extraction**: Uses PyPDF2 to extract text from PDF documents page by page, maintaining page number information for accurate referencing.

**Section Detection**: Implements multiple heuristics to identify document sections:
- Pattern matching for common heading formats (ALL CAPS, numbered, roman numerals)
- NLP-based analysis using spaCy for linguistic patterns
- Length and structure analysis to filter out noise

**Context Extraction**: For each detected section, extracts surrounding context (10 lines) to provide better semantic understanding.

### 2. Semantic Relevance Scoring

**Embedding Generation**: Uses the `all-MiniLM-L6-v2` sentence transformer model to create semantic embeddings for:
- Persona description
- Job-to-be-done task
- Document sections (title + context)

**Relevance Calculation**: Computes cosine similarity between section embeddings and both persona/job embeddings, with weighted combination (70% job relevance, 30% persona relevance).

### 3. Section Ranking and Selection

**Ranking**: Sorts all extracted sections by relevance score in descending order.

**Top-K Selection**: Selects the top 5 most relevant sections across all documents.

**Subsection Analysis**: For top sections, extracts refined subsections using sentence segmentation and intelligent grouping.

### 4. Key Features

- **Multi-document Processing**: Handles 3-10 related PDFs simultaneously
- **Cross-domain Adaptability**: Works with diverse document types (research papers, reports, textbooks)
- **Persona Awareness**: Tailors section extraction based on user role and expertise
- **Task-oriented Prioritization**: Focuses on content relevant to specific job requirements
- **Hierarchical Analysis**: Provides both section-level and subsection-level insights

## Models and Libraries Used

- **spaCy (en_core_web_sm)**: Natural language processing for text analysis and sentence segmentation
- **Sentence Transformers (all-MiniLM-L6-v2)**: Semantic embedding generation for relevance scoring
- **PyPDF2**: PDF text extraction
- **scikit-learn**: Cosine similarity calculations
- **NumPy**: Numerical operations for embeddings

## Performance Characteristics

- **Model Size**: ~200MB (sentence transformer + spaCy model)
- **Processing Time**: ~30-45 seconds for 5-7 documents
- **Memory Usage**: ~500MB peak during processing
- **CPU Only**: No GPU dependencies, optimized for CPU execution

## Building and Running

### Build the Docker Image
```bash
docker build --platform linux/amd64 -t challenge1b:latest .
```

### Run the Solution
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1b:latest
```

### Input Format
Place the following in `/app/input/`:
- `challenge1b_input.json` - Configuration file with documents, persona, and job description
- PDF files referenced in the configuration

### Output Format
The solution generates `challenge1b_output.json` in `/app/output/` containing:
- Metadata (documents, persona, job, timestamp)
- Top 5 ranked sections with importance scores
- Refined subsection analysis

## Technical Implementation Details

### Section Detection Algorithm
1. **Pattern-based Detection**: Uses regex patterns to identify common heading structures
2. **Contextual Analysis**: Analyzes line position, length, and formatting
3. **Linguistic Filtering**: Uses spaCy to filter out non-heading content
4. **Deduplication**: Removes duplicate or overly similar sections

### Relevance Scoring Method
1. **Text Preprocessing**: Combines section title and context for comprehensive analysis
2. **Embedding Generation**: Creates 384-dimensional vectors using sentence transformers
3. **Similarity Calculation**: Computes cosine similarity with persona and job embeddings
4. **Weighted Combination**: Balances persona relevance (30%) and job relevance (70%)

### Subsection Extraction
1. **Sentence Segmentation**: Uses spaCy for accurate sentence boundary detection
2. **Intelligent Grouping**: Groups 2-3 sentences into coherent subsections
3. **Quality Filtering**: Ensures subsections meet minimum length requirements
4. **Context Preservation**: Maintains document and page references

## Error Handling and Robustness

- **File Validation**: Checks for document existence before processing
- **Graceful Degradation**: Continues processing even if some documents fail
- **Memory Management**: Processes documents sequentially to manage memory usage
- **Logging**: Comprehensive logging for debugging and monitoring

## Optimization Considerations

- **Efficient Text Processing**: Processes documents in parallel where possible
- **Model Caching**: Loads models once and reuses for all documents
- **Memory Optimization**: Clears intermediate data structures to manage memory
- **Batch Processing**: Groups similar operations for efficiency