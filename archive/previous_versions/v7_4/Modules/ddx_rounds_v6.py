
# =============================================================================
# DDx Rounds - Complete Working Rounds Module
# =============================================================================

"""
Complete rounds system that preserves ALL dynamic generation and sophisticated features
while fixing execution_time errors and maintaining DDx_Main_Design.md structure.
"""

import time
import json
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum

# Import core components
from ddx_core_v6 import DDxAgent, AgentResponse, SpecialtyType
from ddx_utils import extract_diagnoses, validate_medical_response

# =============================================================================
# 1. Round Framework
# =============================================================================

class RoundType(Enum):
    """All round types - DDx_Main_Design.md + enhancements"""
    # Core DDx_Main_Design.md rounds
    SPECIALIZED_RANKING = "specialized_ranking"
    SYMPTOM_MANAGEMENT = "symptom_management"
    TEAM_INDEPENDENT_DIFFERENTIALS = "team_independent_differentials"
    MASTER_LIST_GENERATION = "master_list_generation"
    REFINEMENT_AND_JUSTIFICATION = "refinement_and_justification"
    POST_DEBATE_VOTING = "post_debate_voting"
    CANT_MISS = "cant_miss"

    # Legacy support
    TRIAGE = "triage"
    SUBSPECIALIST_CONSULTATION = "subspecialist_consultation"
    INDEPENDENT_DIFFERENTIALS = "independent_differentials"
    REFINEMENT_DEBATE = "refinement_debate"
    SYNTHESIS = "synthesis"

@dataclass
class RoundResult:
    """Result of a diagnostic round"""
    round_type: RoundType
    participants: List[str]
    responses: Dict[str, AgentResponse]
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0
    formatted_output: str = ""

class BaseRound(ABC):
    """Base class for all diagnostic rounds"""

    def __init__(self, round_type: RoundType):
        self.round_type = round_type
        self.result = None

    @abstractmethod
    def execute(self, agents: List[DDxAgent], case_description: str,
           context: Dict[str, Any] = None, global_transcript: Dict = None) -> RoundResult:
        """Execute the round with given agents, context, and sliding context transcript"""
        pass

    def _create_result(self, participants: List[str], responses: Dict[str, AgentResponse],
                      metadata: Dict[str, Any] = None, success: bool = True,
                      error: Optional[str] = None, execution_time: float = 0.0,
                      formatted_output: str = "") -> RoundResult:
        """FIXED: Helper to create round result with execution_time support"""
        return RoundResult(
            round_type=self.round_type,
            participants=participants,
            responses=responses,
            metadata=metadata or {},
            success=success,
            error=error,
            execution_time=execution_time,
            formatted_output=formatted_output
        )

# =============================================================================
# 2. Specialized Ranking Round (Dynamic Generation Friendly)
# =============================================================================

class SpecializedRankingRound(BaseRound):
    """Ranking round that works with ANY dynamic specialists"""

    def __init__(self):
        super().__init__(RoundType.SPECIALIZED_RANKING)

    def execute(self, agents: List[DDxAgent], case_description: str,
           context: Dict[str, Any] = None, global_transcript: Dict = None) -> RoundResult:
        """Execute ranking with dynamic specialists"""

        print(f"\n🏥 SPECIALIZED RANKING ROUND")
        print("=" * 50)

        # Use primary agent for ranking
        primary_agent = agents[0] if agents else None
        if not primary_agent:
            return self._create_result([], {}, success=False, error="No agents available")

        # Create dynamic ranking prompt
        specialists_list = [f"- {agent.name}: {agent.specialty.value}" for agent in agents]
        specialists_text = "\n".join(specialists_list)

        ranking_prompt = f"""
Analyze this case and rank the medical specialties by relevance:

CASE:
{case_description}

AVAILABLE SPECIALISTS:
{specialists_text}

Provide your analysis in this format:

<RANK>
1. [Specialty Name] - [Brief justification]
2. [Specialty Name] - [Brief justification]
...
</RANK>

<CAN'T MISS>
- [Critical diagnosis that cannot be missed]
- [Another critical diagnosis]
</CAN'T MISS>

Rank ALL available specialists and provide clear justifications.
"""

        try:
            start_time = time.time()
            response = primary_agent.generate_response(ranking_prompt, "specialized_ranking", global_transcript)
            execution_time = time.time() - start_time

            # Extract rankings and can't miss diagnoses
            ranking_data = self._parse_ranking_response(response.content)

            print(f"✅ Ranking completed by {primary_agent.name}")
            print(f"   🎯 Ranked {len(ranking_data.get('rankings', []))} specialties")
            print(f"   🚨 Identified {len(ranking_data.get('cant_miss', []))} critical diagnoses")

            return self._create_result(
                participants=[primary_agent.name],
                responses={primary_agent.name: response},
                metadata=ranking_data,
                execution_time=execution_time,
                formatted_output=response.content
            )

        except Exception as e:
            return self._create_result(
                participants=[primary_agent.name],
                responses={},
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _parse_ranking_response(self, content: str) -> Dict[str, Any]:
        """FIXED: Flexible ranking response parsing that handles multiple output formats"""
        data = {
            'rankings': [],
            'cant_miss': [],
            'total_ranked': 0
        }

        # Method 1: Try structured XML tags (original format)
        rank_match = re.search(r'<RANK>(.*?)</RANK>', content, re.DOTALL)
        if rank_match:
            rank_text = rank_match.group(1).strip()
            lines = rank_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and re.match(r'^\d+\.', line):
                    data['rankings'].append(line)

        # Method 2: If no XML tags, look for numbered lists (FLEXIBLE FALLBACK)
        if not data['rankings']:
            # Look for any numbered ranking patterns
            numbered_pattern = r'^\s*(\d+[\.\)\-\s]+.+(?:cardiology|nephrology|surgery|medicine|emergency|pulmonology|endocrinology|dermatology|neurology|psychiatry|oncology|radiology).*)$'
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                # Look for lines that contain medical specialties and are numbered
                if re.search(numbered_pattern, line, re.IGNORECASE) and len(line) > 10:
                    data['rankings'].append(line.strip())

        # Method 3: If still empty, extract specialties mentioned with ranking words
        if not data['rankings']:
            # Look for ranking language with specialties
            specialty_ranking_patterns = [
                r'(?:first|primary|most|top|1st).*?(cardiology|nephrology|surgery|emergency|medicine|pulmonology|endocrinology|dermatology|neurology|psychiatry|oncology|radiology)',
                r'(?:second|next|2nd).*?(cardiology|nephrology|surgery|emergency|medicine|pulmonology|endocrinology|dermatology|neurology|psychiatry|oncology|radiology)',
                r'(?:third|3rd).*?(cardiology|nephrology|surgery|emergency|medicine|pulmonology|endocrinology|dermatology|neurology|psychiatry|oncology|radiology)'
            ]

            for i, pattern in enumerate(specialty_ranking_patterns, 1):
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    formatted_ranking = f"{i}. {match.title()} - relevant to case"
                    if formatted_ranking not in data['rankings']:
                        data['rankings'].append(formatted_ranking)

        # Set total count
        data['total_ranked'] = len(data['rankings'])

        # Extract can't miss diagnoses (flexible patterns)
        cantmiss_patterns = [
            r'<CAN\'T MISS>(.*?)</CAN\'T MISS>',
            r'(?:can\'t miss|cannot miss|critical|emergency|urgent):\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'(?:must rule out|rule out|exclude):\s*(.*?)(?:\n\n|\n[A-Z]|$)'
        ]

        for pattern in cantmiss_patterns:
            cantmiss_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if cantmiss_match:
                cantmiss_text = cantmiss_match.group(1).strip()
                lines = cantmiss_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('-') or len(line.split()) >= 2):
                        clean_line = line.lstrip('-').strip()
                        if clean_line and clean_line not in data['cant_miss']:
                            data['cant_miss'].append(clean_line)
                break  # Use first pattern that matches

        # If still no cant_miss, look for medical conditions in the text
        if not data['cant_miss']:
            medical_urgency_patterns = [
                r'\b(myocardial infarction|heart attack|stroke|sepsis|pulmonary embolism|pneumonia|acute kidney injury)\b',
                r'\b(diabetic ketoacidosis|hypoglycemia|anaphylaxis|cardiac arrest|respiratory failure)\b'
            ]

            for pattern in medical_urgency_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.lower() not in [item.lower() for item in data['cant_miss']]:
                        data['cant_miss'].append(match.title())

        # Debug output
        if not data['rankings'] and not data['cant_miss']:
            print(f"⚠️ Ranking parsing failed. Content preview:")
            print(f"   First 200 chars: {content[:200]}...")
            print(f"   Looking for medical specialties and ranking language...")
        else:
            print(f"✅ Extracted {len(data['rankings'])} rankings and {len(data['cant_miss'])} critical items")

        return data

# =============================================================================
# 3. Symptom Management Round
# =============================================================================

class SymptomManagementRound(BaseRound):
    """Immediate symptom management and intervention planning"""

    def __init__(self):
        super().__init__(RoundType.SYMPTOM_MANAGEMENT)

    def execute(self, agents: List[DDxAgent], case_description: str,
           context: Dict[str, Any] = None, global_transcript: Dict = None) -> RoundResult:
        """Execute symptom management"""

        print(f"\n🚨 SYMPTOM MANAGEMENT ROUND")
        print("=" * 40)

        # Select emergency/acute care specialist
        emergency_agent = self._select_emergency_agent(agents)

        symptom_prompt = f"""
Analyze this case for IMMEDIATE symptom management and interventions:

CASE:
{case_description}

Provide immediate management plan in this format:

<SYMPTOM_MITIGATION>
[Symptom 1]: [Immediate intervention]
[Symptom 2]: [Immediate intervention]
...
</SYMPTOM_MITIGATION>

<IMMEDIATE_ACTIONS>
- [Action 1]
- [Action 2]
...
</IMMEDIATE_ACTIONS>

Focus on stabilization and immediate care priorities.
"""

        try:
            start_time = time.time()
            response = emergency_agent.generate_response(symptom_prompt, "symptom_management")
            execution_time = time.time() - start_time

            # Parse symptom management
            mgmt_data = self._parse_symptom_management(response.content)

            print(f"🏥 Symptom management by {emergency_agent.name}")
            print(f"   Identified {mgmt_data.get('immediate_interventions_count', 0)} immediate interventions")

            # Display interventions
            for intervention in mgmt_data.get('interventions', []):
                print(f"   • {intervention}")

            return self._create_result(
                participants=[emergency_agent.name],
                responses={emergency_agent.name: response},
                metadata=mgmt_data,
                execution_time=execution_time,
                formatted_output=response.content
            )

        except Exception as e:
            return self._create_result(
                participants=[emergency_agent.name],
                responses={},
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _select_emergency_agent(self, agents: List[DDxAgent]) -> DDxAgent:
        """Select best agent for emergency management"""
        # Prefer emergency/acute care specialists
        emergency_keywords = ['emergency', 'critical', 'acute', 'intensive', 'trauma']

        for agent in agents:
            specialty_lower = agent.specialty.value.lower()
            if any(keyword in specialty_lower for keyword in emergency_keywords):
                return agent

        # Fallback to first agent
        return agents[0] if agents else None

    def _parse_symptom_management(self, content: str) -> Dict[str, Any]:
        """Parse symptom management response"""
        data = {
            'interventions': [],
            'immediate_actions': [],
            'immediate_interventions_count': 0
        }

        # Extract symptom mitigations
        mitigation_match = re.search(r'<SYMPTOM_MITIGATION>(.*?)</SYMPTOM_MITIGATION>', content, re.DOTALL)
        if mitigation_match:
            mitigation_text = mitigation_match.group(1).strip()
            lines = mitigation_text.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    data['interventions'].append(line)

        # Extract immediate actions
        actions_match = re.search(r'<IMMEDIATE_ACTIONS>(.*?)</IMMEDIATE_ACTIONS>', content, re.DOTALL)
        if actions_match:
            actions_text = actions_match.group(1).strip()
            lines = actions_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and line.startswith('-'):
                    data['immediate_actions'].append(line[1:].strip())

        data['immediate_interventions_count'] = len(data['interventions']) + len(data['immediate_actions'])
        return data

# =============================================================================
# 4. Team Independent Differentials Round
# =============================================================================

class TeamIndependentDifferentialsRound(BaseRound):
    """Each specialist provides independent differential diagnosis"""

    def __init__(self):
        super().__init__(RoundType.TEAM_INDEPENDENT_DIFFERENTIALS)

    def execute(self, agents: List[DDxAgent], case_description: str,
           context: Dict[str, Any] = None, global_transcript: Dict = None) -> RoundResult:
        """Execute independent differentials"""

        print(f"\n🔬 TEAM INDEPENDENT DIFFERENTIALS ROUND")
        print("=" * 50)

        differential_prompt = f"""
As a {'{specialty}'} specialist, provide your independent differential diagnosis for this case:

CASE:
{case_description}

Provide your analysis in this format:

<DIFFERENTIAL_DIAGNOSIS>
1. [Primary diagnosis] - [Confidence: XX%] - [Reasoning]
2. [Secondary diagnosis] - [Confidence: XX%] - [Reasoning]
3. [Tertiary diagnosis] - [Confidence: XX%] - [Reasoning]
</DIFFERENTIAL_DIAGNOSIS>

<SPECIALTY_FOCUS>
[Your specialty-specific insights and considerations]
</SPECIALTY_FOCUS>

Provide 3-5 diagnoses with confidence scores and clear reasoning.
"""

        responses = {}
        all_diagnoses = []

        try:
            start_time = time.time()

            for agent in agents:
                agent_prompt = differential_prompt.format(specialty=agent.specialty.value)
                response = agent.generate_response(agent_prompt, "team_independent_differentials")  # Note: changed to match the condition
                responses[agent.name] = response

                # FIXED: Use the structured_data that was already extracted
                if response.structured_data:
                    diagnoses = [{"diagnosis": k, "evidence": v} for k, v in response.structured_data.items()]
                else:
                    # Fallback to XML parsing for compatibility
                    diagnoses = self._extract_agent_diagnoses(response.content)
                
                all_diagnoses.extend(diagnoses)
                print(f"✅ {agent.name} ({agent.specialty.value})")
                print(f"   🎯 Provided {len(diagnoses)} diagnoses")

            execution_time = time.time() - start_time

            # Compile metadata
            metadata = {
                'total_diagnoses': len(all_diagnoses),
                'agent_count': len(agents),
                'all_diagnoses': all_diagnoses,
                'diagnoses_by_agent': {name: self._extract_agent_diagnoses(resp.content)
                                     for name, resp in responses.items()}
            }

            print(f"\n📊 Round Summary:")
            print(f"   Total diagnoses collected: {metadata['total_diagnoses']}")
            print(f"   Participating agents: {metadata['agent_count']}")

            return self._create_result(
                participants=list(responses.keys()),
                responses=responses,
                metadata=metadata,
                execution_time=execution_time
            )

        except Exception as e:
            return self._create_result(
                participants=list(responses.keys()),
                responses=responses,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _extract_agent_diagnoses(self, content: str) -> List[Dict[str, Any]]:
        """Extract diagnoses from agent response"""
        diagnoses = []

        diff_match = re.search(r'<DIFFERENTIAL_DIAGNOSIS>(.*?)</DIFFERENTIAL_DIAGNOSIS>', content, re.DOTALL)
        if diff_match:
            diff_text = diff_match.group(1).strip()
            lines = diff_text.split('\n')

            for line in lines:
                line = line.strip()
                if line and re.match(r'^\d+\.', line):
                    # Parse diagnosis line
                    diagnosis = self._parse_diagnosis_line(line)
                    if diagnosis:
                        diagnoses.append(diagnosis)

        return diagnoses

    def _parse_diagnosis_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse individual diagnosis line"""
        try:
            # Remove number prefix
            content = re.sub(r'^\d+\.\s*', '', line)

            # Extract confidence if present
            confidence_match = re.search(r'Confidence:\s*(\d+)%', content)
            confidence = int(confidence_match.group(1)) if confidence_match else None

            # Extract diagnosis name (before first dash)
            parts = content.split(' - ')
            diagnosis_name = parts[0].strip() if parts else content.strip()

            return {
                'diagnosis': diagnosis_name,
                'confidence': confidence,
                'reasoning': ' - '.join(parts[1:]) if len(parts) > 1 else '',
                'raw_line': line
            }
        except:
            return None

# =============================================================================
# 5. Master List Generation Round
# =============================================================================

class MasterListGenerationRound(BaseRound):
    """Generate consolidated master diagnosis list"""

    def __init__(self):
        super().__init__(RoundType.MASTER_LIST_GENERATION)

    def execute(self, agents: List[DDxAgent], case_description: str,
           context: Dict[str, Any] = None, global_transcript: Dict = None) -> RoundResult:
        """Execute master list generation"""

        print(f"\n📋 MASTER LIST GENERATION ROUND")
        print("=" * 45)

        # Get previous round results
        previous_differentials = context.get('round_results', {}).get(RoundType.TEAM_INDEPENDENT_DIFFERENTIALS)
        if not previous_differentials:
            return self._create_result([], {}, success=False,
                                     error="No previous differential round found")

        all_diagnoses = previous_differentials.metadata.get('all_diagnoses', [])

        # Select primary consolidator
        primary_agent = agents[0] if agents else None

        master_prompt = f"""
Consolidate all diagnoses from the team into a unified master list:

CASE:
{case_description}

ALL TEAM DIAGNOSES:
{self._format_diagnoses_for_prompt(all_diagnoses)}

Create a consolidated master list in this format:

<MASTER DDX LIST>
1. [Diagnosis] - [Combined confidence] - [Synthesis of reasoning]
...
</MASTER DDX LIST>

<CONSOLIDATION_NOTES>
[Explain how diagnoses were grouped, deduplicated, and prioritized]
</CONSOLIDATION_NOTES>

Remove duplicates, group similar diagnoses, and rank by likelihood.
"""

        try:
            start_time = time.time()
            response = primary_agent.generate_response(master_prompt, "master_list_generation", global_transcript)
            execution_time = time.time() - start_time

            # Parse master list
            master_data = self._parse_master_list(response.content, all_diagnoses)

            print(f"✅ Master list generated by {primary_agent.name}")
            print(f"   📋 Master list: {master_data.get('deduplicated_count', 0)} (from {len(all_diagnoses)})")

            return self._create_result(
                participants=[primary_agent.name],
                responses={primary_agent.name: response},
                metadata=master_data,
                execution_time=execution_time,
                formatted_output=response.content
            )

        except Exception as e:
            return self._create_result(
                participants=[primary_agent.name],
                responses={},
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _format_diagnoses_for_prompt(self, diagnoses: List[Dict[str, Any]]) -> str:
        """Format diagnoses for prompt - SHORTENED to avoid token limit"""
        formatted = []
        for i, diag in enumerate(diagnoses, 1):
            confidence_text = f" ({diag['confidence']}%)" if diag.get('confidence') else ""
            # TRUNCATE reasoning to max 50 characters to avoid massive prompts
            reasoning = diag.get('reasoning', '')
            if len(reasoning) > 50:
                reasoning = reasoning[:50] + "..."
            formatted.append(f"{i}. {diag['diagnosis']}{confidence_text} - {reasoning}")
        return '\n'.join(formatted)

    def _parse_master_list(self, content: str, original_diagnoses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """FIXED: Robust master list response parsing that handles multiple output formats"""
        data = {
            'master_diagnoses': [],
            'consolidation_notes': '',
            'original_diagnosis_count': len(original_diagnoses),
            'deduplicated_count': 0
        }

        # Method 1: Try structured XML tags (original format)
        master_match = re.search(r'<MASTER DDX LIST>(.*?)</MASTER DDX LIST>', content, re.DOTALL)
        if master_match:
            master_text = master_match.group(1).strip()
            lines = master_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and re.match(r'^\d+\.', line):
                    data['master_diagnoses'].append(line)

        # Method 2: If no XML tags, use ddx_utils extract_diagnoses (ROBUST FALLBACK)
        if not data['master_diagnoses']:
            from ddx_utils import extract_diagnoses
            extracted_diagnoses = extract_diagnoses(content)

            # Convert to numbered format for consistency
            for i, diagnosis in enumerate(extracted_diagnoses, 1):
                formatted_diagnosis = f"{i}. {diagnosis.strip()}"
                data['master_diagnoses'].append(formatted_diagnosis)

        # Method 3: If still empty, try manual numbered list extraction (ANY format)
        if not data['master_diagnoses']:
            # Look for any numbered lists in the content
            numbered_pattern = r'^\s*(\d+[\.\)\-\s]+.+)$'
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(numbered_pattern, line) and len(line) > 5:  # Reasonable length
                    # Clean up the line but keep original if it looks good
                    if re.match(r'^\d+\.\s*.+', line):
                        data['master_diagnoses'].append(line.strip())
                    else:
                        # Extract just the diagnosis part
                        cleaned_line = re.sub(r'^\s*\d+[\.\)\-\s]+', '', line).strip()
                        if cleaned_line and len(cleaned_line) > 3:  # Valid diagnosis
                            formatted_diagnosis = f"{len(data['master_diagnoses']) + 1}. {cleaned_line}"
                            data['master_diagnoses'].append(formatted_diagnosis)

        # Method 4: Last resort - extract from JSON-like structures
        if not data['master_diagnoses']:
            # Look for JSON-like structures that might contain diagnoses
            json_pattern = r'\{[^{}]*\}'
            json_matches = re.findall(json_pattern, content)
            for match in json_matches:
                try:
                    import json
                    data_obj = json.loads(match)
                    if isinstance(data_obj, dict):
                        for i, diagnosis in enumerate(data_obj.keys(), 1):
                            formatted_diagnosis = f"{i}. {diagnosis.strip()}"
                            data['master_diagnoses'].append(formatted_diagnosis)
                        break  # Use first valid JSON found
                except:
                    continue

        # Set final count
        data['deduplicated_count'] = len(data['master_diagnoses'])

        # Extract consolidation notes (flexible patterns)
        notes_patterns = [
            r'<CONSOLIDATION_NOTES>(.*?)</CONSOLIDATION_NOTES>',
            r'(?:consolidation|notes?|reasoning):\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'(?:summary|conclusion):\s*(.*?)(?:\n\n|\n[A-Z]|$)'
        ]

        for pattern in notes_patterns:
            notes_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if notes_match:
                data['consolidation_notes'] = notes_match.group(1).strip()
                break

        # Debug output to help diagnose issues
        if not data['master_diagnoses']:
            print(f"⚠️ Master list parsing failed. Content preview:")
            print(f"   First 200 chars: {content[:200]}...")
            print(f"   Content length: {len(content)} characters")
            # Try one more extraction using basic text patterns
            words = content.split()
            medical_terms = [w for w in words if any(term in w.lower()
                            for term in ['syndrome', 'disease', 'injury', 'itis', 'osis', 'pathy'])]
            if medical_terms:
                print(f"   Found medical terms: {medical_terms[:5]}")
        else:
            print(f"✅ Extracted {len(data['master_diagnoses'])} diagnoses from master list")
            for i, diag in enumerate(data['master_diagnoses'][:3], 1):  # Show first 3
                print(f"   {i}. {diag[:50]}{'...' if len(diag) > 50 else ''}")

        return data

# =============================================================================
# 6. Refinement and Justification Round
# =============================================================================

class RefinementAndJustificationRound(BaseRound):
    """Enhanced debate and refinement with preferential voting elements"""

    def __init__(self):
        super().__init__(RoundType.REFINEMENT_AND_JUSTIFICATION)

    def execute(self, agents: List[DDxAgent], case_description: str,
           context: Dict[str, Any] = None, global_transcript: Dict = None) -> RoundResult:
        """Execute refinement and justification"""

        print(f"\n💬 REFINEMENT AND JUSTIFICATION ROUND")
        print("=" * 50)

        # Get master list from previous round
        master_round = context.get('round_results', {}).get(RoundType.MASTER_LIST_GENERATION)
        if not master_round:
            return self._create_result([], {}, success=False,
                                     error="No master list round found")

        master_diagnoses = master_round.metadata.get('master_diagnoses', [])

        refinement_prompt = f"""
Review and refine the master diagnosis list through critical analysis:

CASE:
{case_description}

CURRENT MASTER LIST:
{chr(10).join(master_diagnoses)}

Provide your analysis in this format:

<DIAGNOSTIC_REFINEMENT>
[Your detailed analysis of each diagnosis]
</DIAGNOSTIC_REFINEMENT>

<EVIDENCE_ASSESSMENT>
[Assessment of supporting/contradicting evidence for top diagnoses]
</EVIDENCE_ASSESSMENT>

<REFINED_RECOMMENDATIONS>
1. [Top diagnosis] - [Refined reasoning]
2. [Second diagnosis] - [Refined reasoning]
3. [Third diagnosis] - [Refined reasoning]
</REFINED_RECOMMENDATIONS>

Focus on evidence quality and diagnostic confidence.
"""

        responses = {}
        all_refinements = []

        try:
            start_time = time.time()

            for agent in agents:
                response = agent.generate_response(refinement_prompt, "refinement_justification")
                responses[agent.name] = response

                # Extract refinements
                refinements = self._extract_refinements(response.content)
                all_refinements.extend(refinements)

                print(f"✅ {agent.name} completed refinement analysis")

            execution_time = time.time() - start_time

            # Analyze refinements
            refinement_data = self._analyze_refinements(all_refinements, responses)

            print(f"\n📊 Refinement Summary:")
            print(f"   💬 High-value interactions: {refinement_data.get('high_value_interactions', 0)}")
            print(f"   🎯 Tier 1 diagnoses: {refinement_data.get('tier_1_count', 0)}")

            return self._create_result(
                participants=list(responses.keys()),
                responses=responses,
                metadata=refinement_data,
                execution_time=execution_time
            )

        except Exception as e:
            return self._create_result(
                participants=list(responses.keys()),
                responses=responses,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _extract_refinements(self, content: str) -> List[Dict[str, Any]]:
        """Extract refinement recommendations"""
        refinements = []

        refined_match = re.search(r'<REFINED_RECOMMENDATIONS>(.*?)</REFINED_RECOMMENDATIONS>', content, re.DOTALL)
        if refined_match:
            refined_text = refined_match.group(1).strip()
            lines = refined_text.split('\n')

            for line in lines:
                line = line.strip()
                if line and re.match(r'^\d+\.', line):
                    refinements.append({
                        'recommendation': line,
                        'content': content
                    })

        return refinements

    def _analyze_refinements(self, refinements: List[Dict[str, Any]], responses: Dict[str, AgentResponse]) -> Dict[str, Any]:
        """FIXED: Analyze refinement quality and interactions with flexible parsing"""

        # Extract from actual response content instead of requiring XML
        high_value_interactions = 0
        tier_1_count = 0

        for response in responses.values():
            content = response.content.lower()

            # Count high-value interaction indicators
            high_value_patterns = [
                'evidence', 'support', 'challenge', 'disagree', 'alternative',
                'clinical reasoning', 'risk factor', 'contradict', 'tier',
                'recommend', 'differential', 'assessment', 'analysis'
            ]
            pattern_count = sum(1 for pattern in high_value_patterns if pattern in content)
            if pattern_count >= 3:  # Multiple reasoning terms = high value
                high_value_interactions += 1

            # Count tier 1 mentions (more flexible)
            tier_1_indicators = ['tier 1', 'tier one', 'first tier', 'top diagnosis', 'primary diagnosis']
            if any(indicator in content for indicator in tier_1_indicators):
                tier_1_count += 1

        return {
            'total_refinements': len(refinements),
            'high_value_interactions': high_value_interactions,
            'tier_1_count': tier_1_count,
            'agent_contributions': {name: len(response.content.split())
                                  for name, response in responses.items()},
            'all_refinements': refinements
        }

# =============================================================================
# 7. Post-Debate Voting Round (Preferential Voting)
# =============================================================================

class PostDebateVotingRound(BaseRound):
    """Enhanced voting with preferential ballot system"""

    def __init__(self):
        super().__init__(RoundType.POST_DEBATE_VOTING)

    def execute(self, agents: List[DDxAgent], case_description: str,
           context: Dict[str, Any] = None, global_transcript: Dict = None) -> RoundResult:
        """Execute preferential voting"""

        print(f"\n🗳️ POST-DEBATE VOTING ROUND")
        print("=" * 40)

        # Get refined recommendations
        refinement_round = context.get('round_results', {}).get(RoundType.REFINEMENT_AND_JUSTIFICATION)
        if not refinement_round:
            return self._create_result([], {}, success=False,
                                     error="No refinement round found")

        # Extract all refined diagnoses for voting
        voting_options = self._extract_voting_options(refinement_round)

        voting_prompt = f"""
Cast your preferential vote on the refined diagnoses:

CASE:
{case_description}

VOTING OPTIONS:
{chr(10).join([f"{i+1}. {opt}" for i, opt in enumerate(voting_options)])}

Provide your ranking in this format:

<PREFERENTIAL_VOTE>
1st Choice: [Diagnosis] - [Brief justification]
2nd Choice: [Diagnosis] - [Brief justification]
3rd Choice: [Diagnosis] - [Brief justification]
</PREFERENTIAL_VOTE>

<VOTING_CONFIDENCE>
Overall confidence in top choice: [XX%]
</VOTING_CONFIDENCE>

Rank your top 3 choices with reasoning.
"""

        responses = {}
        all_votes = []

        try:
            start_time = time.time()

            for agent in agents:
                response = agent.generate_response(voting_prompt, "post_debate_voting", global_transcript)
                responses[agent.name] = response

                # Extract vote
                vote = self._extract_vote(response.content, agent.name)
                if vote:
                    all_votes.append(vote)

                print(f"✅ {agent.name} cast preferential vote")

            execution_time = time.time() - start_time

            # Calculate voting results
            voting_results = self._calculate_preferential_results(all_votes, voting_options)

            print(f"\n📊 Voting Results:")
            print(f"   🗳️ Agent rankings: {len(all_votes)}")
            if voting_results.get('winner'):
                print(f"   🏆 Winner: {voting_results['winner']}")

            return self._create_result(
                participants=list(responses.keys()),
                responses=responses,
                metadata=voting_results,
                execution_time=execution_time
            )

        except Exception as e:
            return self._create_result(
                participants=list(responses.keys()),
                responses=responses,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _extract_voting_options(self, refinement_round: RoundResult) -> List[str]:
        """Extract voting options from refinement round"""
        options = []
        all_refinements = refinement_round.metadata.get('all_refinements', [])

        # Extract unique diagnoses
        seen_diagnoses = set()
        for refinement in all_refinements:
            recommendation = refinement.get('recommendation', '')
            # Extract diagnosis name from "1. Diagnosis - reasoning" format
            if recommendation:
                diagnosis = recommendation.split(' - ')[0].strip()
                diagnosis = re.sub(r'^\d+\.\s*', '', diagnosis)
                if diagnosis and diagnosis not in seen_diagnoses:
                    options.append(diagnosis)
                    seen_diagnoses.add(diagnosis)

        return options[:10]  # Limit to top 10 for voting

    def _extract_vote(self, content: str, agent_name: str) -> Optional[Dict[str, Any]]:
        """Extract preferential vote from response - handles both XML and JSON formats"""
        
        # Method 1: Try XML format first
        vote_match = re.search(r'<PREFERENTIAL_VOTE>(.*?)</PREFERENTIAL_VOTE>', content, re.DOTALL)
        if vote_match:
            vote_text = vote_match.group(1).strip()
            lines = vote_text.split('\n')
            
            rankings = []
            for line in lines:
                line = line.strip()
                if 'Choice:' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        choice_content = parts[1].strip()
                        diagnosis = choice_content.split(' - ')[0].strip()
                        rankings.append(diagnosis)
            
            if rankings:
                confidence_match = re.search(r'Overall confidence.*?(\d+)%', content)
                confidence = int(confidence_match.group(1)) if confidence_match else None
                
                return {
                    'agent': agent_name,
                    'rankings': rankings,
                    'confidence': confidence,
                    'raw_content': content
                }
        
        # Method 2: Try to extract from JSON/structured data  
        # Look for ranking patterns in JSON or fallback text
        rankings = []
        
        # Pattern for "1st choice", "2nd choice", etc.
        choice_patterns = [
            r'1st.*?choice.*?[:\-]\s*([^,\n\-]+)',
            r'2nd.*?choice.*?[:\-]\s*([^,\n\-]+)', 
            r'3rd.*?choice.*?[:\-]\s*([^,\n\-]+)',
            r'first.*?choice.*?[:\-]\s*([^,\n\-]+)',
            r'second.*?choice.*?[:\-]\s*([^,\n\-]+)',
            r'third.*?choice.*?[:\-]\s*([^,\n\-]+)'
        ]
        
        for pattern in choice_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                diagnosis = match.group(1).strip().strip('",')
                if diagnosis and len(diagnosis) > 2:
                    rankings.append(diagnosis)
        
        # Method 3: Look for numbered lists
        if not rankings:
            numbered_pattern = r'^\s*(\d+)\.?\s*([^:\n]+)'
            matches = re.findall(numbered_pattern, content, re.MULTILINE)
            for num, diagnosis in matches[:3]:  # Top 3
                diagnosis = diagnosis.strip().strip('",')
                if len(diagnosis) > 2:
                    rankings.append(diagnosis)
        
        if rankings:
            print(f"   🗳️ {agent_name}: Extracted {len(rankings)} vote choices via fallback")
            return {
                'agent': agent_name,
                'rankings': rankings,
                'confidence': None,
                'raw_content': content
            }
        
        print(f"   ❌ {agent_name}: No valid vote extracted")
        return None

    def _calculate_preferential_results(self, votes: List[Dict[str, Any]], options: List[str]) -> Dict[str, Any]:
        """Calculate preferential voting results using Borda count"""
        if not votes:
            return {'error': 'No valid votes'}

        # Borda count scoring (3 points for 1st, 2 for 2nd, 1 for 3rd)
        scores = {}
        for vote in votes:
            rankings = vote.get('rankings', [])
            for i, diagnosis in enumerate(rankings[:3]):  # Top 3 only
                points = 3 - i  # 3, 2, 1 points
                scores[diagnosis] = scores.get(diagnosis, 0) + points

        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'borda_scores': scores,
            'ranked_results': sorted_results,
            'winner': sorted_results[0][0] if sorted_results else None,
            'all_rankings': {vote['agent']: vote['rankings'] for vote in votes},
            'total_votes': len(votes),
            'voting_summary': {diag: score for diag, score in sorted_results[:5]}  # Top 5
        }

# =============================================================================
# 8. Can't Miss Diagnoses Round
# =============================================================================

class CantMissRound(BaseRound):
    """Critical diagnoses that cannot be missed"""

    def __init__(self):
        super().__init__(RoundType.CANT_MISS)

    def execute(self, agents: List[DDxAgent], case_description: str,
           context: Dict[str, Any] = None, global_transcript: Dict = None) -> RoundResult:
        """Execute can't miss analysis"""

        print(f"\n🚨 CAN'T MISS DIAGNOSES ROUND")
        print("=" * 40)

        cant_miss_prompt = f"""
Identify critical diagnoses that CANNOT be missed for this case:

CASE:
{case_description}

Focus on life-threatening or immediately dangerous conditions in this format:

<CANT_MISS_DIAGNOSES>
- [Critical diagnosis] - [Why it can't be missed] - [Risk level: HIGH/MEDIUM]
- [Critical diagnosis] - [Why it can't be missed] - [Risk level: HIGH/MEDIUM]
</CANT_MISS_DIAGNOSES>

<IMMEDIATE_WORKUP>
- [Required test/intervention]
- [Required test/intervention]
</IMMEDIATE_WORKUP>

<TIME_SENSITIVITY>
[How quickly these conditions must be ruled out or treated]
</TIME_SENSITIVITY>

Prioritize conditions that could cause immediate harm if missed.
"""

        responses = {}
        all_critical = []

        try:
            start_time = time.time()

            for agent in agents:
                response = agent.generate_response(cant_miss_prompt, "cant_miss", global_transcript)
                responses[agent.name] = response

                # Extract critical diagnoses
                critical = self._extract_critical_diagnoses(response.content)
                all_critical.extend(critical)

                print(f"✅ {agent.name} identified {len(critical)} critical diagnoses")

            execution_time = time.time() - start_time

            # Consolidate critical diagnoses
            critical_data = self._consolidate_critical_diagnoses(all_critical, responses)

            print(f"\n📊 Critical Analysis:")
            print(f"   🚨 Critical diagnoses: {len(critical_data.get('critical_diagnoses', []))}")

            return self._create_result(
                participants=list(responses.keys()),
                responses=responses,
                metadata=critical_data,
                execution_time=execution_time
            )

        except Exception as e:
            return self._create_result(
                participants=list(responses.keys()),
                responses=responses,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _extract_critical_diagnoses(self, content: str) -> List[Dict[str, Any]]:
        """Extract critical diagnoses from response"""
        critical = []

        cantmiss_match = re.search(r'<CANT_MISS_DIAGNOSES>(.*?)</CANT_MISS_DIAGNOSES>', content, re.DOTALL)
        if cantmiss_match:
            cantmiss_text = cantmiss_match.group(1).strip()
            lines = cantmiss_text.split('\n')

            for line in lines:
                line = line.strip()
                if line and line.startswith('-'):
                    # Parse "- Diagnosis - Reason - Risk level"
                    parts = line[1:].strip().split(' - ')
                    if len(parts) >= 2:
                        diagnosis = parts[0].strip()
                        reason = parts[1].strip() if len(parts) > 1 else ''
                        risk = parts[2].strip() if len(parts) > 2 else 'MEDIUM'

                        critical.append({
                            'diagnosis': diagnosis,
                            'reason': reason,
                            'risk_level': risk,
                            'raw_line': line
                        })

        return critical

    def _consolidate_critical_diagnoses(self, all_critical: List[Dict[str, Any]],
                                      responses: Dict[str, AgentResponse]) -> Dict[str, Any]:
        """Consolidate and prioritize critical diagnoses"""
        # Group by diagnosis name
        diagnosis_groups = {}
        for critical in all_critical:
            diag_name = critical['diagnosis']
            if diag_name not in diagnosis_groups:
                diagnosis_groups[diag_name] = []
            diagnosis_groups[diag_name].append(critical)

        # Prioritize by frequency and risk level
        prioritized = []
        for diag_name, group in diagnosis_groups.items():
            high_risk_count = sum(1 for c in group if 'HIGH' in c.get('risk_level', ''))
            total_mentions = len(group)

            prioritized.append({
                'diagnosis': diag_name,
                'mention_count': total_mentions,
                'high_risk_mentions': high_risk_count,
                'priority_score': total_mentions + (high_risk_count * 2),
                'all_reasons': [c['reason'] for c in group],
                'consensus_risk': 'HIGH' if high_risk_count > 0 else 'MEDIUM'
            })

        # Sort by priority score
        prioritized.sort(key=lambda x: x['priority_score'], reverse=True)

        return {
            'critical_diagnoses': prioritized,
            'total_critical_identified': len(all_critical),
            'unique_critical_diagnoses': len(diagnosis_groups),
            'high_priority_count': len([p for p in prioritized if p['consensus_risk'] == 'HIGH']),
            'agent_contributions': {name: len(self._extract_critical_diagnoses(resp.content))
                                  for name, resp in responses.items()}
        }

# =============================================================================
# 9. Round Orchestrator
# =============================================================================

class RoundOrchestrator:
    """Complete orchestrator for all diagnostic rounds"""

    def __init__(self, ddx_system):
        self.ddx_system = ddx_system

        # Initialize all rounds
        self.rounds = {
            RoundType.SPECIALIZED_RANKING: SpecializedRankingRound(),
            RoundType.SYMPTOM_MANAGEMENT: SymptomManagementRound(),
            RoundType.TEAM_INDEPENDENT_DIFFERENTIALS: TeamIndependentDifferentialsRound(),
            RoundType.MASTER_LIST_GENERATION: MasterListGenerationRound(),
            RoundType.REFINEMENT_AND_JUSTIFICATION: RefinementAndJustificationRound(),
            RoundType.POST_DEBATE_VOTING: PostDebateVotingRound(),
            RoundType.CANT_MISS: CantMissRound()
        }

        self.round_results = {}
        self.full_transcript = {}

    def execute_complete_sequence(self, case_description: str,
                                rounds_to_run: List[RoundType] = None) -> Dict[RoundType, RoundResult]:
        """Execute complete diagnostic sequence"""

        if not self.ddx_system.current_agents:
            raise ValueError("No agents available. Run case analysis first.")

        # Default sequence
        if rounds_to_run is None:
            rounds_to_run = [
                RoundType.SPECIALIZED_RANKING,
                RoundType.SYMPTOM_MANAGEMENT,
                RoundType.TEAM_INDEPENDENT_DIFFERENTIALS,
                RoundType.MASTER_LIST_GENERATION,
                RoundType.REFINEMENT_AND_JUSTIFICATION,
                RoundType.POST_DEBATE_VOTING,
                RoundType.CANT_MISS
            ]

        print(f"\n🔄 EXECUTING COMPLETE DDx SEQUENCE")
        print(f"Rounds planned: {[r.value for r in rounds_to_run]}")
        print("=" * 60)

        agents = self.ddx_system.current_agents
        context = {'round_results': self.round_results}

        # Initialize transcript
        self.full_transcript = {
            'case_description': case_description,
            'agents': [
                {
                    'name': agent.name,
                    'specialty': agent.specialty.value,
                    'persona': agent.persona,
                    'model_id': agent.model_id
                }
                for agent in agents
            ],
            'rounds': {},
            'sequence_metadata': {
                'total_rounds': len(rounds_to_run),
                'start_time': time.time()
            }
        }

        # Execute each round
        for round_type in rounds_to_run:
            print(f"\n▶️ Starting {round_type.value.title().replace('_', ' ')} Round...")

            try:
                round_impl = self.rounds[round_type]
                start_time = time.time()

                # CRITICAL CHANGE: Pass full_transcript to round execution
                result = round_impl.execute(agents, case_description, context, self.full_transcript)
                execution_time = time.time() - start_time
                result.execution_time = execution_time

                self.round_results[round_type] = result

                # Capture transcript
                round_transcript = {
                    'round_type': round_type.value,
                    'participants': result.participants,
                    'responses': {},
                    'metadata': result.metadata,
                    'success': result.success,
                    'execution_time': execution_time,
                    'formatted_output': result.formatted_output
                }

                # Save agent responses with timestamps
                # Save agent responses
                for agent_name, response in result.responses.items():
                    round_transcript['responses'][agent_name] = {
                        'content': response.content,
                        'structured_data': response.structured_data,
                        'response_time': response.response_time,
                        'confidence_score': response.confidence_score,
                        'reasoning_quality': response.reasoning_quality,
                        'timestamp': time.time()
                    }

                self.full_transcript['rounds'][round_type.value] = round_transcript

                if result.success:
                    print(f"✅ {round_type.value.title().replace('_', ' ')} Round completed successfully ({execution_time:.1f}s)")
                    print(f"   🤝 Sliding context: {len(self.full_transcript['rounds'])} previous rounds available")
                    # Update context for next round
                    context['previous_round'] = result
                    context['round_results'] = self.round_results
                else:
                    print(f"❌ {round_type.value.title().replace('_', ' ')} Round failed: {result.error}")

            except Exception as e:
                print(f"❌ {round_type.value.title().replace('_', ' ')} Round crashed: {e}")
                self.round_results[round_type] = RoundResult(
                    round_type=round_type,
                    participants=[],
                    responses={},
                    metadata={},
                    success=False,
                    error=str(e)
                )

        # Complete transcript
        self.full_transcript['sequence_metadata']['end_time'] = time.time()
        self.full_transcript['sequence_metadata']['total_execution_time'] = (
            self.full_transcript['sequence_metadata']['end_time'] -
            self.full_transcript['sequence_metadata']['start_time']
        )

        return self.round_results

    def get_sequence_summary(self) -> str:
        """Generate comprehensive sequence summary"""
        if not self.round_results:
            return "No rounds executed yet."

        total_time = sum(r.execution_time for r in self.round_results.values() if r.execution_time)

        summary = f"\n🎯 DIAGNOSTIC SEQUENCE SUMMARY\n"
        summary += "=" * 50 + "\n"

        for round_type, result in self.round_results.items():
            summary += f"\n▶️ {round_type.value.title().replace('_', ' ')}: "

            if result.success:
                summary += f"✅ Success ({result.execution_time:.1f}s)\n"

                # Add round-specific metrics
                if round_type == RoundType.SPECIALIZED_RANKING:
                    ranked = result.metadata.get('total_ranked', 0)
                    cant_miss = len(result.metadata.get('cant_miss', []))
                    summary += f"   🎯 Ranked specialties: {ranked}\n"
                    summary += f"   🚨 Can't miss identified: {cant_miss}\n"

                elif round_type == RoundType.SYMPTOM_MANAGEMENT:
                    interventions = result.metadata.get('immediate_interventions_count', 0)
                    summary += f"   🚨 Immediate interventions: {interventions}\n"

                elif round_type == RoundType.TEAM_INDEPENDENT_DIFFERENTIALS:
                    total_diag = result.metadata.get('total_diagnoses', 0)
                    summary += f"   🔬 Total diagnoses: {total_diag}\n"

                elif round_type == RoundType.MASTER_LIST_GENERATION:
                    master_count = result.metadata.get('deduplicated_count', 0)
                    original_count = result.metadata.get('original_diagnosis_count', 0)
                    summary += f"   📋 Master list: {master_count} (from {original_count})\n"

                elif round_type == RoundType.REFINEMENT_AND_JUSTIFICATION:
                    high_value = result.metadata.get('high_value_interactions', 0)
                    tier_1 = result.metadata.get('tier_1_count', 0)
                    summary += f"   💬 High-value interactions: {high_value}\n"
                    summary += f"   🎯 Tier 1 diagnoses: {tier_1}\n"

                elif round_type == RoundType.POST_DEBATE_VOTING:
                    rankings = result.metadata.get('all_rankings', {})
                    winner = result.metadata.get('winner', 'None')
                    summary += f"   🗳️ Agent rankings: {len(rankings)}\n"
                    summary += f"   🏆 Winner: {winner}\n"

                elif round_type == RoundType.CANT_MISS:
                    critical = len(result.metadata.get('critical_diagnoses', []))
                    high_priority = result.metadata.get('high_priority_count', 0)
                    summary += f"   🚨 Critical diagnoses: {critical}\n"
                    summary += f"   ⚠️ High priority: {high_priority}\n"

            else:
                summary += f"❌ Failed ({result.error})\n"

        summary += f"\n📊 SEQUENCE TOTALS:\n"
        summary += f"   Total execution time: {total_time:.1f}s\n"
        summary += f"   Successful rounds: {sum(1 for r in self.round_results.values() if r.success)}\n"
        summary += f"   Total agents: {len(self.ddx_system.current_agents)}\n"

        return summary

# =============================================================================
# 10. Integration Functions
# =============================================================================

def integrate_rounds_with_ddx_system():
    """Integrate rounds with DDx system"""

    def add_round_orchestrator(self):
        """Add round orchestrator to DDx system"""
        if not hasattr(self, 'round_orchestrator'):
            self.round_orchestrator = RoundOrchestrator(self)

    def run_diagnostic_sequence(self, rounds_to_run: List[RoundType] = None) -> Dict[RoundType, RoundResult]:
        """Run complete diagnostic sequence"""
        if not hasattr(self, 'round_orchestrator'):
            self.add_round_orchestrator()

        if not self.case_data:
            raise ValueError("No case loaded. Run analyze_case() first.")

        case_description = self.case_data.get('case_description') or self.case_data.get('description', '')
        return self.round_orchestrator.execute_complete_sequence(case_description, rounds_to_run)

    def get_diagnostic_summary(self) -> str:
        """Get diagnostic sequence summary"""
        if not hasattr(self, 'round_orchestrator'):
            return "No diagnostic sequence run yet."

        return self.round_orchestrator.get_sequence_summary()

    # Add methods to DDx system class
    return {
        'add_round_orchestrator': add_round_orchestrator,
        'run_diagnostic_sequence': run_diagnostic_sequence,
        'get_diagnostic_summary': get_diagnostic_summary
    }

# Add integration at module level
__all__ = [
    'RoundType', 'RoundResult', 'BaseRound', 'RoundOrchestrator',
    'SpecializedRankingRound', 'SymptomManagementRound',
    'TeamIndependentDifferentialsRound', 'MasterListGenerationRound',
    'RefinementAndJustificationRound', 'PostDebateVotingRound', 'CantMissRound',
    'integrate_rounds_with_ddx_system'
]
# Apply the integration when module is imported
from ddx_core_v6 import DDxSystem

# Get the integration methods
integration_methods = integrate_rounds_with_ddx_system()

# Add them to DDxSystem
DDxSystem.add_round_orchestrator = integration_methods['add_round_orchestrator']
DDxSystem.run_complete_diagnostic_sequence = integration_methods['run_diagnostic_sequence']
DDxSystem.get_diagnostic_summary = integration_methods['get_diagnostic_summary']
