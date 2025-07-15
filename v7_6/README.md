This directory contains v7.6 modules and transcripts.

**Specs**:
```python

conservative_model:
  name: 'Conservative-Model-Qwen2-FULL'
  model_path: 'Qwen/Qwen2-7B-Instruct'
  quantization: null # This is a full-precision model
  dtype: 'bfloat16'  # Recommended for modern models on A100
  memory_fraction: 0.43 # Allocating 43% of 42GB (~18GB) to each instance
  temperature: 0.2
  top_p: 0.7
  max_tokens: 1536
  max_model_len: 4096
  enforce_eager: false
  max_num_seqs: 2


innovative_model:
  name: 'Innovative-Model-Llama3-GPTQ'
  model_path: 'MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ'
  quantization: 'gptq_marlin'
  dtype: 'half'
  memory_fraction: 0.45
  temperature: 0.9
  top_p: 0.7
  max_tokens: 1536
  max_model_len: 4096
  enforce_eager: false
  max_num_seqs: 2


# Rest unchanged
system:
  max_specialists: 20
  max_agents_per_round: 15
  collaboration_depth: 3
  
rounds:
  enable_subspecialist_consultation: true
  enable_preferential_voting: true
  enable_symptom_management: true
  
evaluation:
  clinical_recall_threshold: 0.85
  precision_threshold: 0.80
  
synthesis:
  enable_credibility_weighting: true
  enable_tempo_scoring: true
```
**Example Pipeline Execution (Case 1)**:

```python
from ddx_runner_v6 import DDxRunner

# 1. Define your case name, description, and ground truth
case_name = "Case 1"

case_description = """
Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus and osteoarthritis of the hips. Prior to admission, his medications were insulin and naproxen. He was also started on aspirin, clopidogrel, and metoprolol after the coronary intervention. His temperature is 38 °C (100.4 °F), pulse is 93/min, and blood pressure is 125/85 mm Hg. Examination shows mottled, reticulated purplish discoloration of the feet. Laboratory studies show: Hemoglobin count 14 g/dL Leukocyte count 16,400/mm3 Segmented neutrophils 56% Eosinophils 11% Lymphocytes 31% Monocytes 2% Platelet count 260,000/mm3 Erythrocyte sedimentation rate 68 mm/h Serum Urea nitrogen 25 mg/dL Creatinine 4.2 mg/dL Renal biopsy shows intravascular spindle-shaped vacuoles. What is the most likely cause of this patient's symptoms?
"""

ground_truth = {'acute kidney injury (AKI)': ['decreased urinary output', 'malaise', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL'], 'contrast-induced nephropathy': ['Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL'], 'drug-induced interstitial nephritis': ['decreased urinary output and malaise', 'eosinophils 11%', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL', 'Renal biopsy shows intravascular spindle-shaped vacuoles'], 'cholesterol embolization': ['mottled, reticulated purplish discoloration of the feet', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL', 'intravascular spindle-shaped vacuoles'], 'diabetic nephropathy': ['decreased urinary output', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL']}


# 2. Create an instance of the runner and initialize the system
runner = DDxRunner()
runner.initialize_system()

# 3. Run the complete case with the new, single method
if runner.is_initialized:
    runner.run_case(
        case_name=case_name,
        case_description=case_description,
        ground_truth=ground_truth
    )
```

This will execute all 7 rounds of the DDx pipeline and generate a JSON output of the transcript and metrics.
