This directory contains the modules for version 8 of the LDDx. Of note:
- Evaluator v8 and v9 work in tandem to provide precise string matching with AI enhanced fallback matching to assess clinical equivalence.
- Synonym matching is a work in progress there may be holes depending on the case. When AI fallback evaluation is triggered, user review of output is essential to ensure safe clinical equivalence scoring.

-> **v8_2_dual_qwen:** This setup deploys synchronous dual Qwen2.5 models to power its agents.

-> **v8_2_llama_qwen:** This setup deploys synchronus llama3 and Qwen2.5 models to power its agents.

-> **ddx_results_analyzer.py:** This script contains metric normalization analysis to interpret the outputs of the LDDx in comparison to industry and academic benchamrks. It serves as a translator between novel and traditional metrics. Use to interpret results. See the "Reports" directory for batched analyses and individual case reports.
