from datetime import datetime
from workflow.state import ResearchState, ResearchStep
from services.openai_service import OpenAIService
from utils.document_utils import create_professional_word_doc

class ReportSynthesizerAgent:
    def __init__(self):
        self.openai_service = OpenAIService()
    
    async def execute(self, state: ResearchState) -> ResearchState:
        """Synthesize final report from sections"""
        try:
            sections = state.get("report_sections", {})
            topic = state["topic"]
            
            if not sections:
                return self._handle_empty_sections(state)
            
            # Create final report
            final_report = await self._synthesize_report(sections, topic, state)
            
            # Calculate processing time
            start_time = state.get("start_time")
            end_time = datetime.now().isoformat()
            processing_time = self._calculate_processing_time(start_time, end_time)
            
            return {
                **state,
                "final_report": final_report,
                "end_time": end_time,
                "processing_time": processing_time,
                "current_step": ResearchStep.REPORT_COMPLETE
            }
            
        except Exception as e:
            return self._handle_error(state, e)
    
    async def _synthesize_report(self, sections: dict, topic: str, state: ResearchState) -> str:
        """Combine sections into cohesive final report"""
        
        # Get research metadata
        quality_score = state.get("quality_score", 0.0)
        source_count = len(state.get("all_sources", []))
        internal_count = len(state.get("internal_sources", []))
        external_count = len(state.get("external_sources", []))
        
        # Create report header
        report_header = self._create_report_header(topic, quality_score, source_count, 
                                                  internal_count, external_count)
        
        # Combine sections in logical order
        report_parts = [report_header]
        
        # Add sections in order
        section_order = [
            ("executive_summary", "Executive Summary"),
            ("introduction", "Introduction"),
            ("key_findings", "Key Findings"),
            ("detailed_analysis", "Detailed Analysis"),
            ("conclusion", "Conclusion"),
            ("references", "References")
        ]
        
        for section_key, section_title in section_order:
            if section_key in sections and sections[section_key]:
                report_parts.append(f"\n# {section_title}\n")
                report_parts.append(sections[section_key])
                report_parts.append("\n")
        
        # Add research metadata
        metadata_section = self._create_metadata_section(state)
        report_parts.append(metadata_section)
        
        # Join all parts
        final_report = "\n".join(report_parts)
        
        # Post-process for consistency and quality
        final_report = await self._post_process_report(final_report, topic)
        
        return final_report
    
    def _create_report_header(self, topic: str, quality_score: float, 
                             source_count: int, internal_count: int, 
                             external_count: int) -> str:
        """Create professional report header"""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        header = f"""
# Research Report: {topic}

**Date:** {current_date}
**Report Quality Score:** {quality_score:.2f}/1.00
**Total Sources:** {source_count} ({internal_count} internal, {external_count} external)

---
        """
        
        return header.strip()
    
    def _create_metadata_section(self, state: ResearchState) -> str:
        """Create research metadata section"""
        quality_assessment = state.get("quality_assessment", {})
        
        metadata = f"""
---

# Research Methodology

## Quality Assessment
- **Source Diversity:** {quality_assessment.get('source_diversity', 0):.2f}
- **Content Coverage:** {quality_assessment.get('content_coverage', 0):.2f}
- **Source Credibility:** {quality_assessment.get('source_credibility', 0):.2f}
- **Overall Quality Score:** {state.get('quality_score', 0):.2f}

## Research Process
- **Research Questions:** {len(state.get('research_questions', []))}
- **Internal Sources Found:** {len(state.get('internal_sources', []))}
- **External Sources Found:** {len(state.get('external_sources', []))}
- **Processing Time:** {state.get('processing_time', 'Unknown')} seconds
- **Research Iterations:** {state.get('iteration_count', 1)}

## Information Gaps
"""
        
        gaps = quality_assessment.get('information_gaps', [])
        if gaps:
            for gap in gaps:
                metadata += f"- {gap}\n"
        else:
            metadata += "- No significant information gaps identified\n"
        
        return metadata
    
    async def _post_process_report(self, report: str, topic: str) -> str:
        """Post-process report for quality and consistency"""
        try:
            prompt = f"""
            Review and improve this research report for consistency, clarity, and professional quality.
            
            Original Report:
            {report[:4000]}...
            
            Instructions:
            - Fix any grammatical errors
            - Ensure consistent formatting
            - Improve transitions between sections
            - Maintain professional tone throughout
            - Keep all factual content unchanged
            - Return the improved version
            
            Improved Report:
            """
            
            # Note: This is optional post-processing
            # For large reports, you might want to process sections separately
            
            if len(report) < 3000:  # Only post-process shorter reports
                response = await self.openai_service.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000
                )
                return response.strip()
            else:
                return report  # Return original for longer reports
                
        except Exception as e:
            print(f"Post-processing failed: {e}")
            return report  # Return original on error
    
    def _calculate_processing_time(self, start_time: str, end_time: str) -> float:
        """Calculate processing time in seconds"""
        try:
            if start_time and end_time:
                start_dt = datetime.fromisoformat(start_time)
                end_dt = datetime.fromisoformat(end_time)
                return (end_dt - start_dt).total_seconds()
        except Exception:
            pass
        return 0.0
    
    def _handle_empty_sections(self, state: ResearchState) -> ResearchState:
        """Handle case with empty sections"""
        topic = state["topic"]
        
        minimal_report = f"""
# Research Report: {topic}

**Date:** {datetime.now().strftime("%B %d, %Y")}
**Status:** Incomplete - Insufficient Data

## Summary
Research on '{topic}' was initiated but could not be completed due to insufficient source material or processing errors.

## Recommendation
Please try the research again with:
- More specific search terms
- Expanded source databases
- Alternative research approaches

## Technical Details
- Quality Score: {state.get('quality_score', 0.0):.2f}
- Sources Found: {len(state.get('all_sources', []))}
- Processing Errors: {len(state.get('errors', []))}
        """
        
        return {
            **state,
            "final_report": minimal_report,
            "end_time": datetime.now().isoformat(),
            "current_step": ResearchStep.REPORT_COMPLETE
        }
    
    def _handle_error(self, state: ResearchState, error: Exception) -> ResearchState:
        """Handle synthesis errors"""
        error_info = {
            "type": "report_synthesis_failed",
            "message": str(error),
            "step": "report_synthesis"
        }
        
        errors = state.get("errors", [])
        errors.append(error_info)
        
        # Create error report
        error_report = f"""
# Research Report: {state['topic']}

**Date:** {datetime.now().strftime("%B %d, %Y")}
**Status:** Failed - Synthesis Error

## Error Information
An error occurred during report synthesis: {str(error)}

## Available Data
- Research Questions: {len(state.get('research_questions', []))}
- Sources Found: {len(state.get('all_sources', []))}
- Sections Generated: {len(state.get('report_sections', {}))}

Please try the research again or contact support if the issue persists.
        """
        
        return {
            **state,
            "final_report": error_report,
            "errors": errors,
            "end_time": datetime.now().isoformat(),
            "current_step": ResearchStep.REPORT_COMPLETE
        }
