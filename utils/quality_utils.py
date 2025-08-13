import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import statistics
from services.openai_service import OpenAIService
import logging

class QualityUtils:
    def __init__(self):
        self.openai_service = OpenAIService()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def assess_text_readability(self, text: str) -> float:
        """Assess text readability using various metrics"""
        try:
            if not text or len(text.strip()) < 10:
                return 0.0
            
            # Basic readability metrics
            sentences = self._count_sentences(text)
            words = self._count_words(text)
            syllables = self._count_syllables(text)
            
            if sentences == 0 or words == 0:
                return 0.0
            
            # Simple readability score
            avg_sentence_length = words / sentences
            avg_syllables_per_word = syllables / words
            
            # Flesch-like scoring (simplified)
            readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-1 range
            normalized_score = max(0, min(100, readability_score)) / 100
            
            return normalized_score
            
        except Exception as e:
            self.logger.warning(f"Readability assessment failed: {e}")
            return 0.5  # Default score
    
    def assess_information_density(self, text: str) -> float:
        """Assess how information-dense the text is"""
        try:
            if not text or len(text.strip()) < 10:
                return 0.0
            
            words = text.split()
            if not words:
                return 0.0
            
            # Count information-bearing elements
            info_indicators = 0
            
            # Numbers and statistics
            info_indicators += len(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
            
            # Dates
            info_indicators += len(re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', text))
            
            # Proper nouns (capitalized words, simplified detection)
            info_indicators += len([w for w in words if w[0].isupper() and len(w) > 1])
            
            # Technical terms (words with specific patterns)
            info_indicators += len(re.findall(r'\b[A-Z]{2,}\b', text))  # Acronyms
            info_indicators += len(re.findall(r'\b\w+ly\b', text))  # Adverbs
            
            # Citations and references
            info_indicators += len(re.findall(r'\[.*?\]|\(.*?\)', text))
            
            # URLs and emails
            info_indicators += len(re.findall(r'http[s]?://\S+|\b\w+@\w+\.\w+', text))
            
            # Normalize by text length
            density_score = min(info_indicators / len(words), 1.0)
            
            return density_score
            
        except Exception as e:
            self.logger.warning(f"Information density assessment failed: {e}")
            return 0.5
    
    def assess_source_credibility(self, source_metadata: Dict[str, Any]) -> float:
        """Assess credibility of a source based on metadata"""
        try:
            credibility_score = 0.5  # Base score
            
            # Domain-based credibility
            url = source_metadata.get("url", "").lower()
            domain = source_metadata.get("domain", "").lower()
            
            # High credibility domains
            high_cred_indicators = [
                ".edu", ".gov", ".org", "wikipedia", "ieee", "acm", 
                "springer", "nature", "science", "pubmed", "scholar.google"
            ]
            
            if any(indicator in url or indicator in domain for indicator in high_cred_indicators):
                credibility_score += 0.3
            
            # Medium credibility indicators
            med_cred_indicators = [
                "research", "university", "institute", "journal", "academic"
            ]
            
            title = source_metadata.get("title", "").lower()
            if any(indicator in title for indicator in med_cred_indicators):
                credibility_score += 0.1
            
            # Negative credibility indicators
            low_cred_indicators = [
                "blog", "forum", "social", "opinion", "advertisement"
            ]
            
            if any(indicator in url or indicator in domain or indicator in title 
                   for indicator in low_cred_indicators):
                credibility_score -= 0.2
            
            # HTTPS bonus
            if url.startswith("https"):
                credibility_score += 0.1
            
            # Content length consideration
            content_length = source_metadata.get("word_count", 0)
            if content_length > 500:
                credibility_score += 0.1
            elif content_length < 100:
                credibility_score -= 0.1
            
            # Date recency (if available)
            pub_date = source_metadata.get("published_date", "")
            if pub_date:
                # Bonus for recent content (simplified)
                credibility_score += 0.05
            
            return max(0.0, min(1.0, credibility_score))
            
        except Exception as e:
            self.logger.warning(f"Source credibility assessment failed: {e}")
            return 0.5
    
    async def assess_content_relevance(self, content: str, query: str) -> float:
        """Assess how relevant content is to a query"""
        try:
            if not content or not query:
                return 0.0
            
            # Simple keyword overlap
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words:
                return 0.0
            
            # Basic overlap score
            overlap = len(query_words.intersection(content_words))
            overlap_score = overlap / len(query_words)
            
            # Enhance with semantic similarity if content is substantial
            if len(content) > 200:
                semantic_score = await self._assess_semantic_relevance(content, query)
                # Combine scores
                relevance_score = (overlap_score * 0.6) + (semantic_score * 0.4)
            else:
                relevance_score = overlap_score
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            self.logger.warning(f"Content relevance assessment failed: {e}")
            return 0.5
    
    async def _assess_semantic_relevance(self, content: str, query: str) -> float:
        """Use LLM to assess semantic relevance"""
        try:
            prompt = f"""
            Rate how relevant this content is to the query on a scale of 0.0 to 1.0:
            
            Query: {query}
            
            Content: {content[:1000]}...
            
            Consider:
            - Topic alignment
            - Information overlap
            - Contextual relevance
            
            Return only a number between 0.0 and 1.0:
            """
            
            response = await self.openai_service.generate_completion(
                prompt, 
                temperature=0.1, 
                max_tokens=10
            )
            
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.5
                
        except Exception as e:
            self.logger.warning(f"Semantic relevance assessment failed: {e}")
            return 0.5
    
    def assess_source_diversity(self, sources: List[Dict[str, Any]]) -> float:
        """Assess diversity of sources"""
        try:
            if not sources:
                return 0.0
            
            # Check domain diversity
            domains = set()
            source_types = set()
            authors = set()
            
            for source in sources:
                metadata = source.get("metadata", {})
                
                # Domain diversity
                domain = metadata.get("domain", "unknown")
                domains.add(domain)
                
                # Source type diversity
                source_type = source.get("source_type", "unknown")
                source_types.add(source_type)
                
                # Author diversity
                author = metadata.get("author", "unknown")
                authors.add(author)
            
            # Calculate diversity scores
            domain_diversity = min(len(domains) / len(sources), 1.0)
            type_diversity = len(source_types) / min(3, len(sources))  # Expect max 3 types
            author_diversity = min(len(authors) / len(sources), 1.0)
            
            # Combined diversity score
            diversity_score = (domain_diversity * 0.4 + 
                             type_diversity * 0.3 + 
                             author_diversity * 0.3)
            
            return min(1.0, diversity_score)
            
        except Exception as e:
            self.logger.warning(f"Source diversity assessment failed: {e}")
            return 0.5
    
    def identify_quality_issues(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Identify potential quality issues in sources"""
        issues = []
        
        try:
            if not sources:
                issues.append("No sources available")
                return issues
            
            # Check source count
            if len(sources) < 3:
                issues.append("Very few sources available")
            
            # Check source diversity
            domains = set(s.get("metadata", {}).get("domain", "unknown") for s in sources)
            if len(domains) == 1:
                issues.append("All sources from same domain")
            
            # Check content quality
            low_quality_count = 0
            for source in sources:
                content = source.get("content", "")
                if len(content) < 100:
                    low_quality_count += 1
            
            if low_quality_count > len(sources) / 2:
                issues.append("Many sources have very short content")
            
            # Check credibility
            low_cred_count = 0
            for source in sources:
                cred_score = source.get("credibility_score", 0.5)
                if cred_score < 0.4:
                    low_cred_count += 1
            
            if low_cred_count > len(sources) / 3:
                issues.append("Many sources have low credibility scores")
            
            # Check relevance
            low_rel_count = 0
            for source in sources:
                rel_score = source.get("relevance_score", 0.5)
                if rel_score < 0.3:
                    low_rel_count += 1
            
            if low_rel_count > len(sources) / 3:
                issues.append("Many sources have low relevance scores")
            
        except Exception as e:
            self.logger.warning(f"Quality issue identification failed: {e}")
            issues.append("Unable to assess quality issues")
        
        return issues
    
    def calculate_overall_quality_score(
        self, 
        sources: List[Dict[str, Any]], 
        coverage_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall quality score for research"""
        try:
            if not sources:
                return 0.0
            
            # Source-based metrics
            avg_credibility = statistics.mean([
                s.get("credibility_score", 0.5) for s in sources
            ])
            
            avg_relevance = statistics.mean([
                s.get("relevance_score", 0.5) for s in sources
            ])
            
            source_diversity = self.assess_source_diversity(sources)
            
            # Coverage metrics
            content_coverage = coverage_metrics.get("content_coverage", 0.5)
            question_coverage = coverage_metrics.get("question_coverage", 0.5)
            
            # Calculate weighted overall score
            overall_score = (
                avg_credibility * 0.25 +
                avg_relevance * 0.25 +
                source_diversity * 0.20 +
                content_coverage * 0.15 +
                question_coverage * 0.15
            )
            
            return min(1.0, overall_score)
            
        except Exception as e:
            self.logger.warning(f"Overall quality score calculation failed: {e}")
            return 0.5
    
    # Helper methods
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        return len(re.findall(r'[.!?]+', text))
    
    def _count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def _count_syllables(self, text: str) -> int:
        """Estimate syllable count (simplified)"""
        words = text.lower().split()
        syllable_count = 0
        
        for word in words:
            # Simple syllable counting
            word = re.sub(r'[^a-z]', '', word)
            if word:
                # Count vowel groups
                vowels = 'aeiouy'
                syllables = len(re.findall(r'[aeiouy]+', word))
                if word.endswith('e'):
                    syllables -= 1
                syllables = max(1, syllables)  # Every word has at least 1 syllable
                syllable_count += syllables
        
        return syllable_count
