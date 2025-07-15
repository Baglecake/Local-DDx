This directory contains v7.6 modules and transcripts.

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
