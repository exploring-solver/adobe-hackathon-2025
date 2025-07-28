    def _extract_smart_title_from_candidates(self, candidates: List, document_info: Dict[str, Any]) -> str:
        """
        Intelligently extract document title by comparing PDF metadata with first heading.
        Prioritizes the most appropriate title based on quality heuristics.
        """
        self.logger.info("--- Starting Smart Title Extraction from Candidates ---")
        
        metadata_title = document_info.get("title", "").strip()
        first_heading_text = ""
        
        self.logger.info(f"PDF Metadata Title: '{metadata_title}'")
        
        # Find the best first heading candidate (highest confidence on first page)
        first_page_candidates = [c for c in candidates if c.page == 1]
        self.logger.info(f"Found {len(first_page_candidates)} candidates on first page")
        
        if first_page_candidates:
            # Sort by position (top first) then by confidence
            sorted_candidates = sorted(
                first_page_candidates,
                key=lambda x: (x.position_ratio, -x.confidence_score)
            )
            
            self.logger.info("Top 3 first page candidates:")
            for i, cand in enumerate(sorted_candidates[:3]):
                self.logger.info(f"  {i+1}. '{cand.text}' (pos: {cand.position_ratio:.3f}, conf: {cand.confidence_score:.3f})")
            
            best_candidate = sorted_candidates[0]
            
            # Only consider it if it's reasonably positioned and has decent confidence
            if best_candidate.position_ratio < 0.3 and best_candidate.confidence_score > 0.3:
                first_heading_text = best_candidate.text.strip()
                self.logger.info(f"Selected first heading candidate: '{first_heading_text}' (qualified)")
            else:
                self.logger.info(f"Best candidate '{best_candidate.text}' did not qualify (pos: {best_candidate.position_ratio:.3f}, conf: {best_candidate.confidence_score:.3f})")
        else:
            self.logger.info("No candidates found on first page")
        
        self.logger.info(f"First Heading Text: '{first_heading_text}'")
        
        # Decision logic: Compare both titles and choose the most appropriate
        
        # Case 1: No metadata title or it's generic/poor quality
        if not metadata_title or self._is_poor_quality_title(metadata_title):
            self.logger.info("CASE 1: Metadata title is empty or poor quality")
            
            if first_heading_text and not self._is_poor_quality_title(first_heading_text):
                self.logger.info(f"✓ DECISION: Using first heading as title: '{first_heading_text}'")
                return first_heading_text
            elif metadata_title:  # Use metadata as fallback even if poor quality
                self.logger.info(f"↻ DECISION: Using metadata title as fallback: '{metadata_title}'")
                return metadata_title
            else:
                fallback_title = self._generate_fallback_title(document_info)
                self.logger.info(f"⚠ DECISION: Using generated fallback title: '{fallback_title}'")
                return fallback_title
        
        # Case 2: No first heading found or it's poor quality
        if not first_heading_text or self._is_poor_quality_title(first_heading_text):
            self.logger.info("CASE 2: First heading is empty or poor quality")
            self.logger.info(f"✓ DECISION: Using metadata title: '{metadata_title}'")
            return metadata_title
        
        # Case 3: Both titles exist - choose the better one
        self.logger.info("CASE 3: Both titles exist - comparing quality")
        
        metadata_score = self._score_title_quality(metadata_title)
        heading_score = self._score_title_quality(first_heading_text)
        
        self.logger.info(f"Title quality scores:")
        self.logger.info(f"  Metadata: {metadata_score:.3f} - '{metadata_title}'")
        self.logger.info(f"  Heading:  {heading_score:.3f} - '{first_heading_text}'")
        
        # Prefer first heading if it's significantly better or if scores are close
        # (since first heading is more likely to be the actual document title)
        score_diff = heading_score - metadata_score
        if heading_score > metadata_score or abs(score_diff) < 0.2:
            self.logger.info(f"✓ DECISION: Using first heading as title (score difference: {score_diff:.3f}): '{first_heading_text}'")
            return first_heading_text
        else:
            self.logger.info(f"✓ DECISION: Using metadata title (score difference: {score_diff:.3f}): '{metadata_title}'")
            return metadata_title

    def _extract_smart_title_from_structured(self, structured_headings: List[Dict], document_info: Dict[str, Any]) -> str:
        """
        Extract title from structured PDF headings with intelligent comparison.
        """
        self.logger.info("--- Starting Smart Title Extraction from Structured Headings ---")
        
        metadata_title = document_info.get("title", "").strip()
        first_structured_text = ""
        
        self.logger.info(f"PDF Metadata Title: '{metadata_title}'")
        
        if structured_headings:
            # Find the first heading (lowest level number, earliest page)
            sorted_headings = sorted(structured_headings, key=lambda x: (x.get("page", 1), x.get("level", 1)))
            self.logger.info(f"Found {len(structured_headings)} structured headings")
            
            if sorted_headings:
                first_heading = sorted_headings[0]
                first_structured_text = first_heading.get("text", "").strip()
                self.logger.info(f"First structured heading: '{first_structured_text}' (page: {first_heading.get('page', 1)}, level: {first_heading.get('level', 1)})")
        else:
            self.logger.info("No structured headings found")
        
        # Apply same intelligent comparison logic
        if not metadata_title or self._is_poor_quality_title(metadata_title):
            self.logger.info("CASE 1: Metadata title is empty or poor quality")
            
            if first_structured_text and not self._is_poor_quality_title(first_structured_text):
                self.logger.info(f"✓ DECISION: Using first structured heading as title: '{first_structured_text}'")
                return first_structured_text
            elif metadata_title:
                self.logger.info(f"↻ DECISION: Using metadata title as fallback: '{metadata_title}'")
                return metadata_title
            else:
                fallback_title = self._generate_fallback_title(document_info)
                self.logger.info(f"⚠ DECISION: Using generated fallback title: '{fallback_title}'")
                return fallback_title
        
        if not first_structured_text or self._is_poor_quality_title(first_structured_text):
            self.logger.info("CASE 2: First structured heading is empty or poor quality")
            self.logger.info(f"✓ DECISION: Using metadata title: '{metadata_title}'")
            return metadata_title
        
        # Compare quality scores
        self.logger.info("CASE 3: Both titles exist - comparing quality")
        
        metadata_score = self._score_title_quality(metadata_title)
        heading_score = self._score_title_quality(first_structured_text)
        
        self.logger.info(f"Title quality scores:")
        self.logger.info(f"  Metadata: {metadata_score:.3f} - '{metadata_title}'")
        self.logger.info(f"  Structured: {heading_score:.3f} - '{first_structured_text}'")
        
        score_diff = heading_score - metadata_score
        if heading_score > metadata_score or abs(score_diff) < 0.2:
            self.logger.info(f"✓ DECISION: Using structured heading as title (score difference: {score_diff:.3f}): '{first_structured_text}'")
            return first_structured_text
        else:
            self.logger.info(f"✓ DECISION: Using metadata title (score difference: {score_diff:.3f}): '{metadata_title}'")
            return metadata_title
   def _select_best_document_title(self, document_info: Dict[str, Any], headings: List[Dict[str, Any]]) -> str:
        """
        UNIFIED method to select the best document title by comparing PDF metadata with first heading.
        This replaces the separate methods and provides a single point of title selection.
        """
        self.logger.info("=== UNIFIED SMART DOCUMENT TITLE SELECTION ===")
        
        metadata_title = document_info.get("title", "").strip()
        first_heading_text = ""
        
        # Extract first heading text (works for both structured and candidate-based headings)
        if headings:
            first_heading = headings[0]
            first_heading_text = first_heading.get("text", "").strip()
            
            # Log heading source and details
            heading_source = first_heading.get("features", {}).get("source", "text_analysis")
            if heading_source == "pdf_structure":
                self.logger.info(f"First heading from structured TOC: '{first_heading_text}' (page: {first_heading.get('page', 1)})")
            else:
                confidence = first_heading.get("confidence", 0.0)
                page = first_heading.get("page", 1)
                self.logger.info(f"First heading from text analysis: '{first_heading_text}' (page: {page}, confidence: {confidence:.3f})")
        
        self.logger.info(f"PDF Metadata Title: '{metadata_title}'")
        self.logger.info(f"First Heading Text: '{first_heading_text}'")
        
        # Case 1: No headings found - fall back to metadata or default
        if not first_heading_text:
            self.logger.info("CASE 1: No headings found")
            if metadata_title and not self._is_poor_quality_title(metadata_title):
                self.logger.info(f"✓ DECISION: Using metadata title: '{metadata_title}'")
                return metadata_title
            else:
                fallback_title = self._generate_fallback_title(document_info)
                self.logger.info(f"⚠ DECISION: Using fallback title: '{fallback_title}'")
                return fallback_title
        
        # Case 2: No metadata title - use first heading
        if not metadata_title:
            self.logger.info("CASE 2: No metadata title found")
            self.logger.info(f"✓ DECISION: Using first heading: '{first_heading_text}'")
            return first_heading_text
        
        # Case 3: Both exist - intelligent comparison
        self.logger.info("CASE 3: Both titles exist - performing intelligent comparison")
        
        # Quality assessment
        metadata_is_poor = self._is_poor_quality_title(metadata_title)
        heading_is_poor = self._is_poor_quality_title(first_heading_text)
        
        self.logger.info(f"Quality assessment:")
        self.logger.info(f"  Metadata title is poor quality: {metadata_is_poor}")
        self.logger.info(f"  First heading is poor quality: {heading_is_poor}")
        
        # Case 3a: Only metadata is poor
        if metadata_is_poor and not heading_is_poor:
            self.logger.info("CASE 3a: Metadata title is poor quality, heading is good")
            self.logger.info(f"✓ DECISION: Using first heading: '{first_heading_text}'")
            return first_heading_text
        
        # Case 3b: Only heading is poor
        if heading_is_poor and not metadata_is_poor:
            self.logger.info("CASE 3b: First heading is poor quality, metadata is good")
            self.logger.info(f"✓ DECISION: Using metadata title: '{metadata_title}'")
            return metadata_title
        
        # Case 3c: Both are poor - choose less poor
        if metadata_is_poor and heading_is_poor:
            self.logger.info("CASE 3c: Both titles are poor quality - choosing less poor option")
            metadata_score = self._score_title_quality(metadata_title)
            heading_score = self._score_title_quality(first_heading_text)
            
            self.logger.info(f"Poor quality scores:")
            self.logger.info(f"  Metadata: {metadata_score:.3f}")
            self.logger.info(f"  Heading: {heading_score:.3f}")
            
            if heading_score >= metadata_score:
                self.logger.info(f"✓ DECISION: First heading less poor: '{first_heading_text}'")
                return first_heading_text
            else:
                self.logger.info(f"✓ DECISION: Metadata title less poor: '{metadata_title}'")
                return metadata_title
        
        # Case 3d: Both are good - detailed comparison
        self.logger.info("CASE 3d: Both titles are good quality - detailed comparison")
        
        metadata_score = self._score_title_quality(metadata_title)
        heading_score = self._score_title_quality(first_heading_text)
        
        self.logger.info(f"Quality scores:")
        self.logger.info(f"  Metadata: {metadata_score:.3f} - '{metadata_title}'")
        self.logger.info(f"  Heading:  {heading_score:.3f} - '{first_heading_text}'")
        
        # Length analysis
        metadata_len = len(metadata_title)
        heading_len = len(first_heading_text)
        length_ratio = heading_len / metadata_len if metadata_len > 0 else float('inf')
        
        self.logger.info(f"Length analysis:")
        self.logger.info(f"  Metadata length: {metadata_len}")
        self.logger.info(f"  Heading length: {heading_len}")
        self.logger.info(f"  Length ratio (heading/metadata): {length_ratio:.2f}")
        
        # Decision logic
        score_diff = heading_score - metadata_score
        
        if heading_score > metadata_score + 0.1:  # Heading significantly better
            self.logger.info(f"✓ DECISION: First heading significantly better (score diff: +{score_diff:.3f}): '{first_heading_text}'")
            return first_heading_text
        elif metadata_score > heading_score + 0.1:  # Metadata significantly better
            self.logger.info(f"✓ DECISION: Metadata title significantly better (score diff: {score_diff:.3f}): '{metadata_title}'")
            return metadata_title
        else:
            # Scores are close - use length and source heuristics
            self.logger.info(f"Scores are close (diff: {score_diff:.3f}) - using additional heuristics")
            
            # For structured headings, trust the structure more
            if headings and headings[0].get("features", {}).get("source") == "pdf_structure":
                # But only if it passed content validation
                if headings[0].get("features", {}).get("validated_against_content", False):
                    self.logger.info(f"✓ DECISION: Trusting validated structured heading: '{first_heading_text}'")
                    return first_heading_text
            
            # Length-based decision
            if length_ratio > 1.5:  # Heading much longer
                self.logger.info(f"✓ DECISION: First heading much longer and descriptive: '{first_heading_text}'")
                return first_heading_text
            elif length_ratio < 0.67:  # Metadata much longer
                self.logger.info(f"✓ DECISION: Metadata title much longer and descriptive: '{metadata_title}'")
                return metadata_title
            else:
                # Default to first heading (content-derived)
                self.logger.info(f"✓ DECISION: Defaulting to first heading (content-derived): '{first_heading_text}'")
                return first_heading_text


    def _is_poor_quality_title(self, title: str) -> bool:
        """
        Check if a title is of poor quality and should be avoided.
        """
        if not title or len(title.strip()) < 3:
            self.logger.debug(f"Title '{title}' is poor quality: too short")
            return True
        
        title_lower = title.lower().strip()
        
        # Poor quality indicators
        poor_indicators = [
            "untitled", "document", "microsoft word", "pdf", "docx", "doc",
            "page 1", "header", "footer", "temp", "draft", "copy of",
            "new document", "blank", "title", "heading", "chapter 1"
        ]
        
        # Check if title is just a poor indicator
        if title_lower in poor_indicators:
            self.logger.debug(f"Title '{title}' is poor quality: matches poor indicator")
            return True
        
        # Check if title starts with poor indicators
        for indicator in poor_indicators:
            if title_lower.startswith(indicator):
                self.logger.debug(f"Title '{title}' is poor quality: starts with '{indicator}'")
                return True
        
        # Check if title is just numbers or very short
        if len(title.strip()) < 5 and not any(c.isalpha() for c in title):
            self.logger.debug(f"Title '{title}' is poor quality: short and no letters")
            return True
        
        # Check if title is all caps and very generic
        if title.isupper() and len(title) < 10:
            self.logger.debug(f"Title '{title}' is poor quality: all caps and short")
            return True
        
        self.logger.debug(f"Title '{title}' passed quality check")
        return False

    def _score_title_quality(self, title: str) -> float:
        """
        Score title quality from 0.0 to 1.0 based on various heuristics.
        Higher score indicates better quality.
        """
        if not title:
            self.logger.debug("Title scoring: Empty title, score = 0.0")
            return 0.0
        
        score = 0.5  # Base score
        title_clean = title.strip()
        title_lower = title_clean.lower()
        
        self.logger.debug(f"Title scoring for: '{title_clean}'")
        self.logger.debug(f"  Base score: {score:.3f}")
        
        # Length scoring (optimal range: 10-60 characters)
        length = len(title_clean)
        if 10 <= length <= 60:
            score += 0.2
            self.logger.debug(f"  Length bonus (+0.2): optimal length {length}")
        elif 5 <= length <= 80:
            score += 0.1
            self.logger.debug(f"  Length bonus (+0.1): acceptable length {length}")
        elif length < 5 or length > 100:
            score -= 0.2
            self.logger.debug(f"  Length penalty (-0.2): poor length {length}")
        
        # Content quality indicators
        
        # Positive indicators
        if any(c.isalpha() for c in title_clean):  # Contains letters
            score += 0.1
            self.logger.debug("  Letter bonus (+0.1): contains letters")
        
        if title_clean.count(' ') >= 1:  # Multi-word title
            score += 0.1
            self.logger.debug("  Multi-word bonus (+0.1): contains spaces")
        
        if title_clean[0].isupper() and not title_clean.isupper():  # Proper case
            score += 0.1
            self.logger.debug("  Proper case bonus (+0.1): starts with capital but not all caps")
        
        if ':' in title_clean and title_clean.count(':') == 1:  # Subtitle
            score += 0.1
            self.logger.debug("  Subtitle bonus (+0.1): contains single colon")
        
        # Negative indicators
        if self._is_poor_quality_title(title_clean):
            score -= 0.4
            self.logger.debug("  Poor quality penalty (-0.4): failed quality check")
        
        if title_clean.startswith(('Chapter', 'Section', 'Part')):  # Structural elements
            score -= 0.2
            self.logger.debug("  Structural penalty (-0.2): starts with structural word")
        
        if title_clean.endswith('.pdf'):  # Filename-like
            score -= 0.3
            self.logger.debug("  Filename penalty (-0.3): ends with .pdf")
        
        if title_clean.isupper():  # All caps (often poor quality)
            score -= 0.1
            self.logger.debug("  All caps penalty (-0.1): all uppercase")
        
        if re.match(r'^\d+\.?\s*$', title_clean):  # Just numbers
            score -= 0.5
            self.logger.debug("  Numbers-only penalty (-0.5): just numbers")
        
        # Domain-specific positive indicators
        academic_indicators = ['analysis', 'study', 'research', 'investigation', 'report']
        for indicator in academic_indicators:
            if indicator in title_lower:
                score += 0.1
                self.logger.debug(f"  Academic bonus (+0.1): contains '{indicator}'")
                break
        
        final_score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
        self.logger.debug(f"  Final score: {final_score:.3f}")
        
        return final_score

    def _generate_fallback_title(self, document_info: Dict[str, Any]) -> str:
        """Generate a fallback title when no good title is found."""
        self.logger.info("--- Generating Fallback Title ---")
        
        filename = document_info.get("filename", "")
        self.logger.info(f"Original filename: '{filename}'")
        
        if filename:
            # Clean up filename
            title = Path(filename).stem
            self.logger.info(f"Filename stem: '{title}'")
            
            title = title.replace('_', ' ').replace('-', ' ')
            self.logger.info(f"After replacing underscores/hyphens: '{title}'")
            
            # Remove common prefixes
            prefixes_to_remove = ["Microsoft Word - ", "PDF - ", "Document - "]
            for prefix in prefixes_to_remove:
                if title.startswith(prefix):
                    title = title[len(prefix):]
                    self.logger.info(f"After removing prefix '{prefix}': '{title}'")
                    break
            
            final_title = title if title else "Untitled Document"
            self.logger.info(f"Generated fallback title: '{final_title}'")
            return final_title
        
        self.logger.info("No filename available, using default: 'Untitled Document'")
        return "Untitled Document"
    
    def _analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF document and extract metadata."""
        
        # Basic validation
        if not validate_pdf(pdf_path):
            raise ValueError(f"Invalid PDF file: {pdf_path}")
        
        file_size = os.path.getsize(pdf_path)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            self.logger.warning(f"Large file size: {file_size / (1024*1024):.1f}MB")
        
        document_info = {
            "filename": Path(pdf_path).name,
            "file_path": str(pdf_path),
            "file_size": file_size,
            "processing_method": "hybrid"
        }
        
        # Extract PDF metadata using PyMuPDF
        try:
            with fitz.open(pdf_path) as doc:
                metadata = doc.metadata
                document_info.update({
                    "total_pages": len(doc),
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", ""),
                })
                
                # Detect language if set to auto
                if self.language == 'auto':
                    detected_lang = self._detect_document_language(doc)
                    document_info["language"] = detected_lang
                    self.language = detected_lang
                else:
                    document_info["language"] = self.language
                
                # Analyze document structure (only if metadata will be used)
                if self._should_include_metadata():
                    structure_info = self._analyze_document_structure(doc)
                    document_info.update(structure_info)
                
        except Exception as e:
            self.logger.warning(f"Failed to extract PDF metadata: {e}")
            document_info.update({
                "total_pages": 0,
                "language": self.language,
            })
        
        return document_info'