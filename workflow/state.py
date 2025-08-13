from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum

class ResearchStep(str, Enum):
    INITIALIZED = "initialized"
    PLANNING = "planning"
    PLANNING_COMPLETE = "planning_complete"
    INTERNAL_RESEARCH = "internal_research"
    EXTERNAL_RESEARCH = "external_research"
    RESEARCH_COMPLETE = "research_complete"
    QUALITY_ASSESSMENT = "quality_assessment"
    QUALITY_ASSESSED = "quality_assessed"
    REPORT_GENERATION = "report_generation"
    SECTIONS_GENERATED = "sections_generated"
    REPORT_SYNTHESIS = "report_synthesis"
    REPORT_COMPLETE = "report_complete"
    ERROR_RECOVERY = "error_recovery"

class Source(TypedDict):
    content: str
    title: str
    url: Optional[str]
    source_type: str  # 'internal' or 'external'
    relevance_score: float
    credibility_score: Optional[float]
    question: str
    metadata: Dict[str, Any]

class QualityAssessment(TypedDict):
    source_diversity: float
    content_coverage: float
    source_credibility: float
    information_gaps: List[str]
    overall_score: float

class ResearchState(TypedDict):
    # Input
    topic: str
    
    # Planning
    research_plan: Optional[Dict[str, Any]]
    research_questions: List[str]
    
    # Research Results
    internal_sources: List[Source]
    external_sources: List[Source]
    all_sources: List[Source]
    
    # Quality Assessment
    quality_assessment: Optional[QualityAssessment]
    quality_score: float
    
    # Report Generation
    report_sections: Dict[str, str]
    final_report: Optional[str]
    
    # State Management
    current_step: ResearchStep
    iteration_count: int
    errors: List[Dict[str, Any]]
    
    # Metadata
    start_time: Optional[str]
    end_time: Optional[str]
    processing_time: Optional[float]
