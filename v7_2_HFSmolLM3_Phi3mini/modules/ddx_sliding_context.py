
# =============================================================================
# ddx_sliding_context.py - Sliding Context Window for Agent Collaboration
# =============================================================================

"""
Implementation of sliding context window for enhanced agent collaboration.
This enables agents to see and respond to each other's reasoning, creating
true epistemic labor division and collaborative emergence.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

# =============================================================================
# Context Filtering and Management
# =============================================================================

@dataclass
class ContextEntry:
    """Single entry in the sliding context window"""
    round_type: str
    agent_name: str
    content: str
    confidence_score: float
    reasoning_quality: str
    timestamp: float
    is_high_value: bool = False
    contradicts_agent: Optional[str] = None

class ContextFilterType(Enum):
    """Types of context filtering"""
    RECENT_ROUNDS = "recent_rounds"           # Last N rounds
    HIGH_CONFIDENCE = "high_confidence"       # High confidence responses
    OPPOSING_VIEWS = "opposing_views"         # Contradictory positions
    KEY_EXCHANGES = "key_exchanges"           # Important interactions
    SPECIALIST_RELEVANT = "specialist_relevant" # Relevant to current agent

class SlidingContextManager:
    """Manages sliding context window for agent interactions"""

    def __init__(self, max_context_length: int = 1500):
        self.max_context_length = max_context_length
        self.context_cache = {}

    def build_context_for_agent(self, agent_name: str, agent_specialty: str,
                               round_type: str, full_transcript: Dict,
                               filter_types: List[ContextFilterType] = None) -> str:
        """Build contextual summary for an agent's prompt"""

        if not full_transcript or 'rounds' not in full_transcript:
            return ""

        # Default filtering strategy
        if filter_types is None:
            filter_types = self._get_default_filters(round_type)

        # Extract relevant context entries
        context_entries = self._extract_context_entries(full_transcript)

        # Apply filters
        filtered_entries = self._apply_filters(
            context_entries, agent_name, agent_specialty, filter_types
        )

        # Build context string with attention guidance
        context_string = self._build_context_string(
            filtered_entries, round_type, agent_name
        )

        # Ensure within length limits
        return self._truncate_context(context_string)

    def _get_default_filters(self, round_type: str) -> List[ContextFilterType]:
        """Get default filtering strategy based on round type"""

        filter_map = {
            'specialized_ranking': [ContextFilterType.RECENT_ROUNDS],
            'symptom_management': [ContextFilterType.RECENT_ROUNDS],
            'team_independent_differentials': [ContextFilterType.RECENT_ROUNDS],
            'master_list_generation': [
                ContextFilterType.HIGH_CONFIDENCE,
                ContextFilterType.SPECIALIST_RELEVANT
            ],
            'refinement_and_justification': [
                ContextFilterType.OPPOSING_VIEWS,
                ContextFilterType.HIGH_CONFIDENCE,
                ContextFilterType.KEY_EXCHANGES
            ],
            'post_debate_voting': [
                ContextFilterType.KEY_EXCHANGES,
                ContextFilterType.HIGH_CONFIDENCE
            ],
            'cant_miss': [
                ContextFilterType.OPPOSING_VIEWS,
                ContextFilterType.HIGH_CONFIDENCE
            ]
        }

        return filter_map.get(round_type, [ContextFilterType.RECENT_ROUNDS])

    def _extract_context_entries(self, full_transcript: Dict) -> List[ContextEntry]:
        """Extract all context entries from transcript"""
        entries = []

        for round_name, round_data in full_transcript['rounds'].items():
            if 'responses' not in round_data:
                continue

            for agent_name, response_data in round_data['responses'].items():
                entry = ContextEntry(
                    round_type=round_name,
                    agent_name=agent_name,
                    content=response_data.get('content', ''),
                    confidence_score=response_data.get('confidence_score', 0.5),
                    reasoning_quality=response_data.get('reasoning_quality', 'standard'),
                    timestamp=response_data.get('timestamp', 0),
                    is_high_value=self._assess_high_value(response_data)
                )
                entries.append(entry)

        return entries

    def _assess_high_value(self, response_data: Dict) -> bool:
        """Assess if a response is high-value for context"""
        content = response_data.get('content', '').lower()
        confidence = response_data.get('confidence_score', 0.5)

        # High-value indicators
        value_indicators = [
            'evidence', 'contradict', 'disagree', 'alternative',
            'however', 'but', 'challenge', 'oppose', 'support',
            'clinical reasoning', 'risk factor', 'unlikely',
            'more likely', 'less probable', 'consider'
        ]

        indicator_count = sum(1 for indicator in value_indicators if indicator in content)

        # High value if: high confidence + reasoning markers, or many reasoning markers
        return (confidence > 0.7 and indicator_count >= 2) or indicator_count >= 4

    def _apply_filters(self, entries: List[ContextEntry], agent_name: str,
                      agent_specialty: str, filter_types: List[ContextFilterType]) -> List[ContextEntry]:
        """Apply filtering strategies to context entries"""

        filtered = entries.copy()

        for filter_type in filter_types:
            if filter_type == ContextFilterType.RECENT_ROUNDS:
                filtered = self._filter_recent_rounds(filtered)
            elif filter_type == ContextFilterType.HIGH_CONFIDENCE:
                filtered = self._filter_high_confidence(filtered)
            elif filter_type == ContextFilterType.OPPOSING_VIEWS:
                filtered = self._filter_opposing_views(filtered, agent_name)
            elif filter_type == ContextFilterType.SPECIALIST_RELEVANT:
                filtered = self._filter_specialist_relevant(filtered, agent_specialty)
            elif filter_type == ContextFilterType.KEY_EXCHANGES:
                filtered = self._filter_key_exchanges(filtered)

        return filtered

    def _filter_recent_rounds(self, entries: List[ContextEntry]) -> List[ContextEntry]:
        """Filter to recent rounds only"""
        # Sort by timestamp (most recent first) and limit count
        entries.sort(key=lambda x: x.timestamp, reverse=True)
        return entries[:8]  # Max 8 entries

    def _filter_high_confidence(self, entries: List[ContextEntry]) -> List[ContextEntry]:
        """Filter to high confidence responses"""
        return [entry for entry in entries if entry.confidence_score > 0.7]

    def _filter_opposing_views(self, entries: List[ContextEntry], agent_name: str) -> List[ContextEntry]:
        """Filter to opposing or contradictory views"""
        opposing = []
        for entry in entries:
            if entry.agent_name != agent_name and entry.is_high_value:
                # Look for contradiction indicators
                content_lower = entry.content.lower()
                if any(word in content_lower for word in ['disagree', 'contradict', 'however', 'alternative']):
                    opposing.append(entry)
        return opposing

    def _filter_specialist_relevant(self, entries: List[ContextEntry], agent_specialty: str) -> List[ContextEntry]:
        """Filter to specialist-relevant content"""
        keywords = self._get_specialty_keywords(agent_specialty)
        relevant = []

        for entry in entries:
            content_lower = entry.content.lower()
            if any(keyword in content_lower for keyword in keywords):
                relevant.append(entry)

        return relevant

    def _filter_key_exchanges(self, entries: List[ContextEntry]) -> List[ContextEntry]:
        """Filter to key exchanges and debates"""
        return [entry for entry in entries if entry.is_high_value or entry.confidence_score > 0.8]

    def _get_specialty_keywords(self, specialty: str) -> List[str]:
        """Get keywords relevant to a medical specialty"""
        specialty_map = {
            'cardiology': ['heart', 'cardiac', 'coronary', 'myocardial', 'ecg', 'chest pain'],
            'pulmonology': ['lung', 'respiratory', 'breathing', 'cough', 'dyspnea', 'pneumonia'],
            'endocrinology': ['diabetes', 'glucose', 'insulin', 'hormone', 'thyroid', 'metabolic'],
            'nephrology': ['kidney', 'renal', 'creatinine', 'urine', 'dialysis', 'nephropathy'],
            'gastroenterology': ['stomach', 'liver', 'intestinal', 'digestive', 'nausea', 'abdominal'],
            'neurology': ['brain', 'neurological', 'seizure', 'cognitive', 'nerve', 'headache'],
            'hematology': ['blood', 'anemia', 'bleeding', 'clotting', 'hemoglobin', 'platelet'],
            'emergency medicine': ['acute', 'emergency', 'critical', 'immediate', 'urgent', 'stabilization']
        }

        specialty_lower = specialty.lower()
        for spec, keywords in specialty_map.items():
            if spec in specialty_lower:
                return keywords

        return ['clinical', 'medical', 'patient', 'diagnosis']  # General keywords

    def _build_context_string(self, entries: List[ContextEntry],
                             round_type: str, agent_name: str) -> str:
        """Build the context string with attention guidance"""

        if not entries:
            return ""

        # Round-specific attention guidance
        attention_guides = {
            'refinement_and_justification':
                "Focus especially on statements that contradict your prior reasoning, "
                "or introduce new evidence. You may reinforce, challenge, or synthesize these perspectives.",
            'post_debate_voting':
                "Consider which arguments were most compelling and well-supported by evidence.",
            'cant_miss':
                "Pay attention to any critical conditions that others may have overlooked.",
            'master_list_generation':
                "Integrate insights from specialists while maintaining clinical accuracy."
        }

        context_parts = []

        # Add attention guidance
        guidance = attention_guides.get(round_type,
            "Build upon or constructively challenge the previous discourse.")
        context_parts.append(f"COLLABORATION GUIDANCE: {guidance}\n")

        # Add recent team discourse
        context_parts.append("RECENT TEAM DISCOURSE:")

        for entry in entries:
            # Truncate long content
            content = entry.content[:200] + "..." if len(entry.content) > 200 else entry.content

            # Add confidence indicator
            confidence_indicator = "🔥" if entry.confidence_score > 0.8 else "⚡" if entry.is_high_value else ""

            context_parts.append(
                f"\n{entry.round_type.upper()} — {entry.agent_name}{confidence_indicator}:\n{content}\n"
            )

        return "\n".join(context_parts)

    def _truncate_context(self, context_string: str) -> str:
        """Ensure context stays within length limits"""
        if len(context_string) <= self.max_context_length:
            return context_string

        # Truncate but try to keep complete entries
        truncated = context_string[:self.max_context_length]

        # Find last complete entry boundary
        last_boundary = truncated.rfind('\n\n')
        if last_boundary > self.max_context_length * 0.7:  # If we can keep 70%+
            truncated = truncated[:last_boundary]

        return truncated + "\n[... context truncated for length ...]"

# =============================================================================
# Testing Function
# =============================================================================

def test_sliding_context():
    """Test the sliding context window functionality"""

    # Mock transcript data
    mock_transcript = {
        'rounds': {
            'independent_differentials': {
                'responses': {
                    'Dr. Smith': {
                        'content': 'I believe this is acute myocardial infarction based on ST elevation and chest pain.',
                        'confidence_score': 0.9,
                        'reasoning_quality': 'high',
                        'timestamp': 1000
                    },
                    'Dr. Jones': {
                        'content': 'I disagree with MI. The symptoms suggest pulmonary embolism instead.',
                        'confidence_score': 0.8,
                        'reasoning_quality': 'high',
                        'timestamp': 1001
                    }
                }
            },
            'refinement_debate': {
                'responses': {
                    'Dr. Smith': {
                        'content': 'However, the ECG changes clearly support my MI diagnosis.',
                        'confidence_score': 0.85,
                        'reasoning_quality': 'standard',
                        'timestamp': 1002
                    }
                }
            }
        }
    }

    # Test context manager
    context_manager = SlidingContextManager()

    context = context_manager.build_context_for_agent(
        agent_name="Dr. Wilson",
        agent_specialty="Cardiology",
        round_type="refinement_and_justification",
        full_transcript=mock_transcript
    )

    print("🧪 Testing Sliding Context Window")
    print("=" * 50)
    print("Generated Context:")
    print(context)
    print("\n✅ Context generation successful!")

    return context

if __name__ == "__main__":
    # Test the sliding context functionality
    test_sliding_context()
