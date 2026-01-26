# Changelog

All notable changes to Local-DDx are documented in this file.

## [v9] - 2025-01

### Added
- Ollama inference backend for Apple Silicon compatibility
- Gradio web interface for interactive demonstrations
- Streamlined 3-round diagnostic workflow (POC)
- Backend abstraction layer supporting future MLX integration

### Changed
- Simplified architecture for portable deployment
- Removed CUDA/vLLM dependency for demonstration version

---

## [v8.6] - 2025-01

### Added
- Comorbidity considerations to encourage specialist collaboration
- Enhanced interactivity prompting between agents
- Multi-condition likelihood assessment

### Metrics Example
```
Interactive Debate Summary:
   Direct challenges: 9
   Position changes: 9
   Evidence citations: 95
   Interaction quality: 7.28
   Comorbidity discoveries: 9
   Comorbidity consensus: strong
   Multi-condition likelihood: HIGH
```

## [v8.5] - 2025-01

### Added
- Devil's Advocate mechanism for diagnostic stress-testing
- Emergent behaviour detection in bounded environments
- Enhanced debate round prompting

### Changed
- Devil's Advocate assigned to most credible specialist
- Refinement round exhibits emergent discourse patterns

## [v8.4] - 2024-12

### Added
- Direct challenges and responses between agents in debate rounds
- Position change tracking through debate
- Agent position changes inform preferential voting

### Changed
- Synchronous Gemma2-Qwen2.5 architecture deployed
- Debate round refactored for genuine interaction

## [v8.3] - 2024-12

### Added
- Master key vulnerability safeguards (per Zhao et al., 2025)
- Consolidated hybrid evaluation module

### Changed
- Production-ready evaluation pipeline
- String matching + AI assessment hybrid approach

## [v8.2] - 2024-11

### Added
- First production-ready v8 iteration
- Llama3 + Qwen2.5 synchronous pairing

## [v8.1] - 2024-11

### Added
- Master key vulnerability checks
- Performance analysis script (ddx_results_analyzer.py)
- Industry/academy metrics translation

### Security
- Hardened AI-enhanced evaluation against "hacking"
- Lenient assessment prevention

---

## [v7.7] - 2024-10

### Changed
- Refined evaluation pipeline for precise ground truth matching
- Tested on dual Qwen2.5-7B-Instruct-GPTQ-Int4

## [v7.6] - 2024-10

### Added
- First mixed-precision model iteration
- Meta-Llama-3-8B-Instruct-GPTQ + Qwen2-7B-Instruct pairing

## [v7.5] - 2024-09

### Changed
- Enhanced parsing and prompt control
- "Chattiness" reduction in agent responses

## [v7.4] - 2024-09

### Added
- Flexible parsing for various output formats

## [v7.3] - 2024-08

### Added
- Dual Llama3 quantized model support

## [v7.2] - 2024-08

### Added
- Production-ready platform (eager: false)
- Interchangeable models via config.yaml

### Changed
- Full vLLM library compatibility

## [v7.1] - 2024-07

### Added
- Stable HuggingFace SmolLM3 + Microsoft Phi3mini support

---

## [v6.0] - 2024-06

### Added
- Initial dual-model scaffolding
- Dynamic agent generation capabilities
- Dual NousResearch/Hermes-2-Pro-Mistral-7B architecture

### Notes
- Developmental version (eager: true)
- Foundation for all subsequent versions
