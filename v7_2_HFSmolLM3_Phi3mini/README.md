This directory contains the modules anf transcripts for the local DDx v7.2
- This version is an extension of v7.1 with "eager" set to false for optimized performance.
- This experiment uses the Hugging Face SmolLM3 and the Microsoft Phi3 Mini models to empower its agents.

**Specs:**

>conservative_model:
>>name: 'Conservative-Model-HFTB/SmolLM3-3B'
>>
>>model_path: 'HuggingFaceTB/SmolLM3-3B'
>>  
>>memory_fraction: 0.35
>>
>>temperature: 0.1
>>
>>top_p: 0.7
>>
>>max_tokens: 1536
>>
>>max_model_len: 4096
>>
>>dtype: 'auto'
>>
>>enforce_eager: false - # 'false' for optimized performance, 'true' for stable development -> for GPU runtimes
>>
>>max_num_seqs: 2

>innovative_model:
>
>>name: 'Innovative-Model-Phi'
>>
>>model_path: 'microsoft/Phi-3-mini-4k-instruct'
>>  
>>memory_fraction: 0.35
>>
>>temperature: 0.9
>>
>>top_p: 0.7
>>
>>max_tokens: 1536
>>
>>max_model_len: 4096
>>dtype: 'auto'
>>enforce_eager: false
>>max_num_seqs: 2

