# LLM-LOCAL-LAB - CLAUDE CODE INSTRUCTIONS

## PROJECT OVERVIEW

LLM-Local-Lab is a professional experimentation and benchmarking repository for running Large Language Models locally on high-performance hardware. This project leverages advanced Claude Code features (subagents, hooks, workflows) to automate benchmarking, analysis, and documentation.

**Primary Goals:**
- Systematic benchmarking of LLMs on local hardware
- Knowledge repository of model configurations and performance
- Automated analysis and documentation generation
- Experimentation with dual-GPU setups, quantization, and optimization

---

## HARDWARE CONTEXT

**System Specifications:**
- **CPU:** AMD Ryzen 9 9950X (16 cores / 32 threads)
- **RAM:** 126 GB DDR5
- **GPUs:** 2x NVIDIA GeForce RTX 5090 (32GB VRAM each, 64GB total)
- **Compute Capability:** 12.0 (latest generation)
- **CUDA:** 13.0
- **PyTorch:** 2.9.0.dev20250829+cu128 (dual-GPU configured)
- **Storage:** 3.63 TB available

**Performance Baseline:**
- Embedding throughput: 2,398 docs/second (batch processing)
- GPU acceleration: 6.9x faster than CPU
- Capable of running 70B Q4 models across dual-GPU
- Optimal for 8B-13B models at FP16 with maximum speed

---

## SUBAGENT USAGE GUIDE

This project uses 4 specialized subagents located in `.claude/agents/`:

### 1. @benchmark-analyst
**When to use:**
- After executing benchmarks to analyze results
- To compare metrics across different model configurations
- To generate performance insights and recommendations
- To detect patterns or anomalies in benchmark data

**Example invocation:**
```
"Analyze the latest benchmark results for Llama 3.1 70B"
"Compare throughput metrics between Q4 and Q5 quantization"
```

### 2. @model-configurator
**When to use:**
- When adding a new model to the registry
- To optimize configuration for dual-GPU setup
- To troubleshoot OOM (out of memory) errors
- To suggest optimal parameters for specific hardware

**Example invocation:**
```
"Configure Mixtral 8x7B for dual RTX 5090 setup"
"What's the optimal batch size for Llama 70B Q4?"
```

### 3. @documentation-writer
**When to use:**
- After completing experiments to update docs
- To generate weekly/monthly reports
- To update models-registry.md with new findings
- To create experiment summaries

**Example invocation:**
```
"Document the results of today's quantization experiments"
"Generate a weekly report of all benchmarks run"
```

### 4. @gpu-optimizer
**When to use:**
- Before running large benchmarks
- To diagnose performance bottlenecks
- To configure parallelization strategies
- To optimize VRAM usage

**Example invocation:**
```
"Check if GPUs are ready for benchmark"
"Suggest optimal GPU allocation for dual-model inference"
```

---

## HOOK BEHAVIORS

Active hooks in this project (configured in `.claude/hooks/`):

### Pre-Benchmark Hook
**Trigger:** Before executing benchmark scripts
**Actions:**
- Verifies GPUs are idle (<10% utilization)
- Clears CUDA cache
- Checks disk space for logs
- Records benchmark start timestamp

**Blocking:** Yes - prevents execution if GPUs are busy

### Post-Benchmark Hook
**Trigger:** After benchmark completion
**Actions:**
- Saves results to `benchmarks/results/raw/`
- Generates metrics snapshot
- Triggers @benchmark-analyst for initial analysis
- Creates auto-commit with standardized format

### Auto-Documentation Hook
**Trigger:** Session end
**Actions:**
- Updates docs/CHANGELOG.md
- Generates session summary
- Updates models-registry.md if new models were tested

---

## BENCHMARKING PROTOCOLS

### Standard Benchmark Workflow

1. **Pre-Flight Check**
   ```bash
   nvidia-smi  # Verify GPUs are idle
   python -c "import torch; torch.cuda.empty_cache()"  # Clear cache
   ```

2. **Execute Benchmark**
   - Use TodoWrite to track multi-step benchmarks
   - Follow naming convention: `YYYYMMDD_HHMMSS_model-name_config.json`
   - Log all parameters in YAML config

3. **Post-Processing**
   - Let post-benchmark hook save results
   - Delegate analysis to @benchmark-analyst
   - Update documentation via @documentation-writer

### Metrics to Capture

**Performance Metrics:**
- Tokens per second (throughput)
- Time to first token (TTFT)
- Total inference time
- Memory usage (VRAM peak and average)

**Quality Metrics:**
- Perplexity (when applicable)
- Output consistency across runs
- Context handling capability

**Resource Metrics:**
- GPU utilization (%)
- Power consumption
- Temperature
- Multi-GPU efficiency

---

## EXPERIMENT WORKFLOWS

### Workflow 1: Testing a New Model

```
1. User requests to test a specific model
2. Delegate to @model-configurator to generate optimal config
3. Create YAML config in models/configs/
4. Pre-benchmark hook validates environment
5. Execute benchmark with TodoWrite tracking
6. Post-benchmark hook saves results
7. @benchmark-analyst generates performance report
8. @documentation-writer updates models-registry.md
9. Git commit with format: [BENCHMARK]: model-name results
```

### Workflow 2: Quantization Comparison

```
1. User requests comparison of quantization levels
2. Create experiment directory in experiments/quantization-comparison/
3. @model-configurator generates configs for each quantization
4. Loop through each config:
   - Execute benchmark
   - Save results
5. @benchmark-analyst compares all metrics
6. @documentation-writer generates comparative report
7. Git commit with format: [EXPERIMENT]: quantization comparison
```

### Workflow 3: Weekly Report Generation

```
1. Trigger: Manual or scheduled (Friday EOD)
2. @benchmark-analyst reviews all benchmarks from the week
3. Generate insights: best-performing models, optimal configs
4. @documentation-writer creates docs/weekly-reports/YYYYMMDD.md
5. Optional: Create GitHub issue with highlights
```

---

## SAFETY RULES

### RULE 1: GPU Validation Before Benchmarks
**ALWAYS verify GPUs are idle before running benchmarks**
- Check `nvidia-smi` output
- Ensure no processes are using GPU
- Clear CUDA cache

### RULE 2: Use TodoWrite for Multi-Step Tasks
**ANY experiment with >3 steps MUST use TodoWrite**
- Track progress of each benchmark
- Only ONE task in_progress at a time
- Mark completed immediately after finishing

### RULE 3: Standardized Naming
**Follow naming conventions:**
- Benchmarks: `YYYYMMDD_HHMMSS_model-name_config.json`
- Reports: `report_YYYYMMDD_experiment-type.md`
- Commits: `[BENCHMARK|EXPERIMENT|DOCS]: description`
- Configs: `model-name-variant.yaml` (e.g., `llama-3.1-70b-q4.yaml`)

### RULE 4: Documentation is Mandatory
**Every experiment MUST be documented:**
- Add entry to docs/models/models-registry.md
- Significant findings → generate report
- Update README if methodology changes

### RULE 5: No Unicode in Python Code
**Python scripts MUST use ASCII-only characters**
- Avoid emojis, special symbols in code
- Use ASCII for variable names, comments, prints
- Always include UTF-8 encoding setup function

```python
def setup_utf8_encoding():
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
```

### RULE 6: Git Commits are Atomic
**Each commit should represent one logical change:**
- Benchmark results → separate commit
- Documentation updates → separate commit
- Code changes → separate commit

---

## COMMON COMMANDS

### Environment Setup
```bash
# Activate project environment
cd C:\Users\Gamer\Dev\LLM-Local-Lab

# Verify PyTorch + CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check GPU status
nvidia-smi

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
```

### Benchmark Execution
```bash
# Run embedding benchmark
python benchmarks/scripts/embedding_benchmark.py

# Run LLM inference benchmark
python benchmarks/scripts/llm_inference_benchmark.py

# Run full benchmark suite
bash scripts/run_benchmark_suite.sh
```

### Analysis and Reporting
```bash
# Generate report from latest results
python utils/report_generator.py --latest

# Compare two benchmark runs
python utils/report_generator.py --compare run1.json run2.json
```

### Subagent Invocations
```
# In Claude Code conversation:
"@benchmark-analyst analyze latest results"
"@model-configurator optimize config for Llama 70B"
"@documentation-writer update registry with new model"
"@gpu-optimizer check system readiness"
```

---

## FILE STRUCTURE REFERENCE

```
.claude/agents/          # Specialized subagent definitions
.claude/hooks/           # Automation hooks
benchmarks/scripts/      # Benchmark execution scripts
benchmarks/configs/      # Benchmark configuration templates
benchmarks/results/      # Raw data and generated reports
models/configs/          # Model-specific YAML configurations
models/loaders/          # Scripts for loading models (Ollama, vLLM, etc.)
experiments/             # Organized experiments by category
utils/                   # Reusable utilities (GPU, logging, metrics)
docs/hardware/           # Hardware specifications and setup guides
docs/models/             # Model registry and configuration guides
docs/benchmarks/         # Benchmarking methodology
docs/workflows/          # Workflow and agent usage documentation
```

---

## DEVELOPMENT GUIDELINES

### Adding a New Benchmark Script
1. Create in `benchmarks/scripts/`
2. Include UTF-8 encoding setup
3. Output results as JSON to `benchmarks/results/raw/`
4. Log to both console and file
5. Include docstring with usage examples
6. Update `docs/benchmarks/methodology.md`

### Adding a New Model Configuration
1. Create YAML in `models/configs/model-name.yaml`
2. Include: model_id, quantization, GPU allocation, batch_size, context_length
3. Document in `docs/models/models-registry.md`
4. Test with @model-configurator validation

### Creating a New Experiment
1. Create directory in `experiments/experiment-type/`
2. Include README.md with hypothesis and methodology
3. Use TodoWrite to track experiment steps
4. Delegate analysis to @benchmark-analyst
5. Generate final report via @documentation-writer

---

## BEST PRACTICES

1. **Always Plan Before Execute**
   - For Tier 2+ tasks, present plan first
   - Wait for user approval
   - Use TodoWrite for tracking

2. **Leverage Subagents**
   - Don't manually analyze benchmark JSONs → use @benchmark-analyst
   - Don't guess optimal configs → ask @model-configurator
   - Don't write docs manually → delegate to @documentation-writer

3. **Trust the Hooks**
   - Pre-benchmark hook prevents invalid runs
   - Post-benchmark hook handles logging automatically
   - Don't manually commit results (hook does it)

4. **Document as You Go**
   - Every benchmark → entry in registry
   - Every insight → update relevant doc
   - Every week → summary report

5. **Version Control Everything**
   - Configs, scripts, results (summaries), docs
   - Atomic commits with standard format
   - Meaningful commit messages

---

## TROUBLESHOOTING

### OOM (Out of Memory) Errors
1. Check current VRAM usage: `nvidia-smi`
2. Reduce batch size in model config
3. Consider lower quantization (Q4 instead of Q5)
4. Consult @model-configurator for optimization

### Poor Performance
1. Verify GPUs are not throttling: check temps in `nvidia-smi`
2. Ensure dual-GPU parallelization is active
3. Check if running on correct CUDA device
4. Ask @gpu-optimizer for diagnostic

### Benchmark Failures
1. Check pre-benchmark hook output for warnings
2. Verify model files are downloaded and accessible
3. Validate YAML config syntax
4. Review logs in `benchmarks/results/raw/`

### Subagent Not Responding
1. Verify subagent exists in `.claude/agents/`
2. Check YAML frontmatter is valid
3. Ensure tools permissions are correct
4. Re-invoke with explicit command

---

## CHANGELOG

- **2025-10-02**: Initial project setup with Claude Code advanced features
  - Created directory structure
  - Configured 4 specialized subagents
  - Implemented 3 automation hooks
  - Migrated embedding benchmark from Langextract
  - Documented hardware specifications (AMD 9950X + dual RTX 5090)
