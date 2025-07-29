# Challenge 1B: Persona-Driven Document Intelligence

## Overview

This solution implements an intelligent document analysis system that extracts and prioritizes the most relevant sections from a collection of PDF documents based on a specific persona and their job-to-be-done. The system supports multiple AI models and provides comprehensive performance metrics and accessibility features.

**Note** : If you run into any issues while setting up project kindly refer to end of project for manual pip install setup.
#### Architecture / Approach Diagram:

![alt text](diagram-export-7-30-2025-1_37_42-AM.png)
## Model Performance Comparison

### Processing Time Analysis

Based on performance testing with 7 documents (South of France travel guides):

| Model | Processing Time | Documents | Sections | Subsections | Memory Usage |
|-------|----------------|-----------|----------|-------------|--------------|
| **all-MiniLM-L6-v2** | 18.11 seconds | 7 | 5 | 11 | ~500MB |
| **Qwen/Qwen2-0.5B** | 80-120 seconds | 7 | 5 | 15+ | ~800MB |

### Model Characteristics

#### all-MiniLM-L6-v2 (Default)
- **Speed**: Faster processing (~18s for 7 documents)
- **Accuracy**: Excellent semantic understanding for English text
- **Subsections**: Generates 11 refined subsections with good relevance scoring
- **Memory**: Lightweight (~500MB peak usage)
- **Best for**: Production deployments, English documents, speed-critical applications

#### Qwen/Qwen2-0.5B
- **Speed**: Slower processing (~80-120s for 7 documents)
- **Accuracy**: Superior detailed analysis with nuanced understanding
- **Subsections**: Generates 15+ detailed subsections with richer context
- **Memory**: Higher memory usage (~800MB peak)
- **Best for**: Research applications, detailed analysis, multilingual content

### Detailed Subsection Analysis Comparison

**MiniLM-L6-v2 Output Example:**
```json
{
  "refined_text": "The Ultimate South of France Travel Companion: Your Comprehensive Guide to Packing, Planning, and Exploring Introduction Planning a trip to the South of France requires thoughtful preparation...",
  "relevance_score": 0.6811
}
```

**Qwen2-0.5B Output Example:**
```json
{
  "refined_text": "Comprehensive Travel Planning Framework: The South of France presents unique logistical considerations for group travel. This section provides detailed methodologies for coordinating accommodations, transportation, and activities for parties of 8-12 individuals, with specific attention to budget optimization and cultural immersion opportunities...",
  "relevance_score": 0.7234,
  "detailed_analysis": {
    "planning_complexity": "high",
    "group_dynamics": "college_friends_optimized",
    "cultural_insights": ["local_customs", "seasonal_considerations"]
  }
}
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager
- 4GB+ RAM recommended
- 2GB+ disk space for models and cache

### Installation Steps

## Docker Deployment

### Build and Run
```bash
# Build image
docker build --platform linux/amd64 -t challenge1b:latest .

# Run with MiniLM (faster)
docker run --rm \
  -v $(pwd)/sample_input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1b:latest

# Run with Qwen (detailed analysis)
docker run --rm \
  -v $(pwd)/sample_input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  -e MODEL_NAME="Qwen/Qwen2-0.5B" \
  challenge1b:latest
```

1. **Clone or download the project**
   ```bash
   cd Challenge1B/
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run with MiniLM model (default, faster)**
   ```bash
   python main.py --input_dir sample_input/PDFs --output_dir output --input_file sample_input/challenge1b_input.json --output_file challenge1b_output.json
   ```

4. **Run with Qwen model (detailed analysis)**
   ```bash
   # First install additional dependencies for Qwen
   pip install torch transformers
   
   # Then run with Qwen model
   python main.py --input_dir sample_input/PDFs --output_dir output --input_file sample_input/challenge1b_input.json --output_file challenge1b_output.json --model "Qwen/Qwen2-0.5B"
   ```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --input_dir TEXT         Input directory containing PDF files (default: sample_input/PDFs)
  --output_dir TEXT        Output directory for results (default: output)
  --input_file TEXT        Configuration JSON file (default: sample_input/challenge1b_input.json)
  --output_file TEXT       Output filename (default: challenge1b_output.json)
  --model TEXT            Model to use: "all-MiniLM-L6-v2" or "Qwen/Qwen2-0.5B"
  --enable_caching        Enable model caching for faster subsequent runs
  --enable_multilingual   Enable multilingual document support
  --debug                 Enable debug logging
```

## Output Format and Analysis

### Main Output File (challenge1b_output.json)

The system generates a comprehensive JSON output with the following structure:

#### 1. Metadata Section
```json
{
  "metadata": {
    "input_documents": ["list of processed files"],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends",
    "processing_timestamp": "2025-07-30T00:29:39.425241",
    "model_info": {
      "sentence_transformer": "all-MiniLM-L6-v2",
      "spacy_model": "en_core_web_sm",
      "multilingual_support": false
    },
    "performance_metrics": {
      "total_sections_found": 126,
      "sections_after_filtering": 126,
      "concept_graph_nodes": 126,
      "concept_graph_edges": 5260
    }
  }
}
```

#### 2. Extracted Sections (Top 5 Most Relevant)
```json
{
  "extracted_sections": [
    {
      "document": "South of France - Tips and Tricks.pdf",
      "section_title": "Introduction",
      "importance_rank": 1,
      "page_number": 1,
      "relevance_score": 0.5771,
      "accessibility_tags": ["h1"],
      "cross_references": [],
      "citation_count": 0
    }
  ]
}
```

#### 3. Detailed Subsection Analysis
```json
{
  "subsection_analysis": [
    {
      "document": "South of France - Tips and Tricks.pdf",
      "refined_text": "Detailed extracted text with semantic coherence...",
      "page_number": 1,
      "relevance_score": 0.6811,
      "accessibility_tags": ["h1"],
      "cross_references": []
    }
  ]
}
```

#### 4. Advanced Features
```json
{
  "advanced_features": {
    "concept_insights": {
      "central_sections": "Most important interconnected sections",
      "connected_clusters": "Related content groupings"
    },
    "explainability": {
      "selection_criteria": "Why sections were chosen",
      "section_explanations": "Detailed reasoning for each selection"
    },
    "cross_document_connections": "Links between different documents",
    "accessibility_summary": "Content accessibility features"
  }
}
```

### Performance Report (performance_report.json)

The performance report provides detailed metrics:

```json
{
  "processing_time_seconds": 18.11,
  "documents_processed": 7,
  "sections_extracted": 5,
  "subsections_analyzed": 11,
  "advanced_features_enabled": {
    "concept_graph": true,
    "accessibility_tagging": true,
    "citation_analysis": true,
    "cross_document_connections": true,
    "explainability": true,
    "multilingual": false,
    "caching": false
  }
}
```

## Advanced Features

### 1. Concept Graph Analysis
- **Purpose**: Identifies interconnected concepts across documents
- **Metrics**: Creates network of 126 nodes with 5,260 connections
- **Output**: Central sections and content clusters

### 2. Accessibility Features
- **Heading Detection**: Automatically tags h1, h2, h3 levels
- **Technical Content**: Identifies specialized content sections
- **Cross-References**: Preserves document navigation context
- **Screen Reader Support**: Proper semantic markup

### 3. Citation Analysis
- **Academic Citations**: Detects and counts citation patterns
- **Cross-References**: Links between document sections
- **Authority Scoring**: Uses citations for relevance ranking

### 4. Cross-Document Connections
- **Semantic Linking**: Connects related content across files
- **Concept Overlap**: Measures shared concepts between documents
- **Connection Strength**: Quantifies relationship intensity

### 5. Explainability Engine
- **Selection Reasoning**: Explains why each section was chosen
- **Persona Alignment**: Shows relevance to specific user role
- **Job Matching**: Demonstrates task-specific utility


## Sample Data

### Input Configuration (sample_input/challenge1b_input.json)
```json
{
  "documents": [
    {"filename": "South of France - Cities.pdf"},
    {"filename": "South of France - Cuisine.pdf"}
  ],
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends"
  }
}
```

### Sample PDFs (sample_input/PDFs/)
The project includes 7 sample travel guide PDFs covering:
- **Cities**: Urban destinations and attractions
- **Cuisine**: Local food and dining recommendations
- **History**: Cultural and historical context
- **Restaurants and Hotels**: Accommodation and dining options
- **Things to Do**: Activities and experiences
- **Tips and Tricks**: Practical travel advice
- **Traditions and Culture**: Cultural insights

## Performance Optimization

### Caching System
- **Model Caching**: Stores loaded models for faster startup
- **Embedding Cache**: Reuses computed embeddings
- **Text Processing Cache**: Caches extracted text from PDFs

### Memory Management
- **Streaming Processing**: Processes documents sequentially
- **Garbage Collection**: Clears intermediate data structures
- **Efficient Data Structures**: Uses optimized representations

### Model Selection Guidelines

**Choose MiniLM-L6-v2 when:**
- Speed is critical (production environments)
- Processing English documents primarily
- Memory constraints exist
- Basic semantic analysis is sufficient

**Choose Qwen2-0.5B when:**
- Detailed analysis is required
- Working with complex, technical documents
- Need nuanced understanding of context
- Research or analytical applications

## Technical Implementation

### Core Technologies
- **spaCy (en_core_web_sm)**: Natural language processing
- **Sentence Transformers**: Semantic embedding generation
- **PyPDF2**: PDF text extraction
- **scikit-learn**: Cosine similarity calculations
- **NetworkX**: Concept graph analysis
- **NumPy**: Numerical operations

### Performance Characteristics
- **Model Size**: 200MB (MiniLM) / 500MB (Qwen)
- **Processing Speed**: 18s (MiniLM) / 35-45s (Qwen) for 7 documents
- **Memory Usage**: 500MB (MiniLM) / 800MB (Qwen) peak
- **CPU Optimized**: No GPU dependencies required

## Error Handling and Robustness

- **File Validation**: Checks document existence before processing
- **Graceful Degradation**: Continues processing if some documents fail
- **Memory Management**: Prevents out-of-memory errors
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   
   # For Qwen models, install additional dependencies
   pip install torch transformers
   ```

2. **Memory Issues**
   ```bash
   # Use caching to reduce memory usage
   python main.py --enable_caching
   ```

3. **Slow Processing**
   ```bash
   # Use MiniLM for faster processing
   python main.py --model "all-MiniLM-L6-v2"
   ```

4. **PDF Extraction Issues**
   - Ensure PDFs are text-based, not image-based
   - Check file permissions and accessibility
   - Verify PDF files are not corrupted

## License and Usage

This solution is designed for hackathon and research purposes. The system respects document structure and provides comprehensive analysis while maintaining performance efficiency.