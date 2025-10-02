# LLM-Local-Lab

![CI Status](https://github.com/YOURUSERNAME/LLM-Local-Lab/workflows/CI%20-%20Code%20Validation/badge.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.9+-red.svg)
![CUDA](https://img.shields.io/badge/cuda-13.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Professional experimentation and benchmarking repository for running Large Language Models on local high-performance hardware.

## Overview

LLM-Local-Lab is a systematic approach to testing, benchmarking, and optimizing LLMs on local infrastructure. This repository leverages advanced Claude Code features (subagents, hooks, automated workflows) to streamline experimentation and knowledge management.

**Key Features:**
- 🚀 Automated benchmarking with pre/post validation hooks
- 🤖 Specialized AI subagents for analysis, configuration, and documentation
- 📊 Structured metrics collection and performance tracking
- 🔧 Hardware-optimized configurations for dual-GPU setups
- 📚 Comprehensive knowledge base of tested models and findings

## Hardware Specifications

This lab runs on high-end consumer hardware optimized for LLM inference:

- **CPU**: AMD Ryzen 9 9950X (16 cores / 32 threads)
- **RAM**: 126 GB DDR5
- **GPUs**: 2x NVIDIA GeForce RTX 5090 (32GB VRAM each, 64GB total)
- **Compute Capability**: 12.0 (latest generation)
- **CUDA**: 13.0
- **PyTorch**: 2.9.0.dev20250829+cu128 (dual-GPU configured)
- **Storage**: 3.63 TB available

**Performance Baseline:**
- Embedding throughput: **2,398 documents/second** (batch processing)
- GPU acceleration: **6.9x faster** than CPU-only inference
- Capable of running **70B parameter models** (Q4 quantization) across dual-GPU
- Optimal for **8B-13B models** at FP16 precision with maximum speed

## Quick Start

### Setup

```bash
# Clone or navigate to repository
cd C:\Users\Gamer\Dev\LLM-Local-Lab

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch + CUDA setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check GPU status
nvidia-smi
```

### Running Your First Benchmark

```bash
# Run embedding benchmark (migrated from Langextract)
python benchmarks/scripts/embedding_benchmark.py

# Results will be saved to benchmarks/results/raw/
```

### Using Subagents

This project includes 4 specialized AI agents accessible via Claude Code:

```
@benchmark-analyst    # Analyze benchmark results, generate insights
@model-configurator   # Create optimal configs for models
@documentation-writer # Generate and update documentation
@gpu-optimizer        # Diagnose performance issues, optimize GPU usage
```

**Example usage:**
```
"@model-configurator optimize config for Llama 3.1 70B"
"@benchmark-analyst compare the last 3 benchmark runs"
"@gpu-optimizer check if system is ready for benchmark"
```

## Project Structure

```
LLM-Local-Lab/
├── .claude/                    # Claude Code configuration
│   ├── agents/                 # Specialized subagents
│   └── hooks/                  # Automation scripts
├── benchmarks/
│   ├── scripts/                # Benchmark execution scripts
│   ├── configs/                # Test configurations
│   └── results/                # Raw data and reports
├── models/
│   ├── configs/                # Model-specific YAML configs
│   ├── loaders/                # Model loading scripts
│   └── quantization/           # Quantization utilities
├── experiments/                # Organized experiments
│   ├── dual-gpu/
│   ├── quantization-comparison/
│   ├── context-length/
│   └── fine-tuning/
├── utils/                      # Reusable utilities
├── docs/                       # Documentation
│   ├── hardware/               # Hardware specs and setup
│   ├── models/                 # Model registry
│   ├── benchmarks/             # Methodology docs
│   └── workflows/              # Usage guides
└── scripts/                    # Automation scripts
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Full guide for Claude Code integration and workflows
- **[Hardware Specs](docs/hardware/system-specs.md)** - Detailed system specifications
- **[Models Registry](docs/models/models-registry.md)** - Tested models and configurations
- **[Agent Usage Guide](docs/workflows/agent-usage.md)** - How to use specialized subagents

## Benchmarking Workflow

### Standard Benchmark Process

1. **Pre-Flight Check**: `@gpu-optimizer` validates system readiness
2. **Configuration**: `@model-configurator` generates optimal config
3. **Execution**: Run benchmark script with automated pre/post hooks
4. **Analysis**: `@benchmark-analyst` generates performance report
5. **Documentation**: `@documentation-writer` updates registry and docs
6. **Commit**: Auto-commit results with standardized format

### Automated Hooks

**Pre-Benchmark Hook**:
- Validates GPUs are idle (<10% utilization)
- Clears CUDA cache
- Checks disk space and temperatures

**Post-Benchmark Hook**:
- Saves results to standardized location
- Generates metrics snapshot
- Logs completion
- Suggests next actions

**Auto-Commit Hook**:
- Commits results with descriptive message
- Includes key metrics in commit message
- Maintains clean git history

## Key Metrics Tracked

**Performance:**
- Throughput (tokens/second or docs/second)
- Latency (time to first token, total inference time)
- Batch processing efficiency

**Resources:**
- VRAM usage (peak and average per GPU)
- GPU utilization percentage
- Multi-GPU load balancing
- Power consumption and temperature

**Quality:**
- Output consistency across runs
- Context handling capability
- Quantization impact on quality

## Model Capabilities

Based on dual RTX 5090 setup (64GB total VRAM):

| Model Size | Quantization | Configuration | Status |
|------------|--------------|---------------|--------|
| 1B-3B | FP16 | Single GPU | ✅ Optimal |
| 7B-13B | FP16/Q8 | Single GPU | ✅ Optimal |
| 30B-34B | Q5/Q4 | Single or Dual GPU | ✅ Good |
| 70B | Q4 | Dual GPU (required) | ✅ Tested |
| 100B+ | Q4/Q3 | Dual GPU | 🧪 Experimental |

## Common Commands

```bash
# Check system status
nvidia-smi
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

# Run benchmarks
python benchmarks/scripts/embedding_benchmark.py
python benchmarks/scripts/llm_inference_benchmark.py

# Generate reports
python utils/report_generator.py --latest
```

## Contributing

This is a personal research repository, but methodology and configurations are documented for reproducibility.

### Adding a New Model Config

1. Create YAML in `models/configs/model-name.yaml`
2. Test with `@model-configurator` validation
3. Run benchmark
4. Document in `docs/models/models-registry.md`

### Adding a New Benchmark

1. Create script in `benchmarks/scripts/`
2. Output JSON to `benchmarks/results/raw/`
3. Update `docs/benchmarks/methodology.md`
4. Run and delegate analysis to `@benchmark-analyst`

## Roadmap

- [ ] Test Llama 3.1 70B Q4 dual-GPU performance
- [ ] Compare Q4 vs Q5 vs FP16 quantization across models
- [ ] Implement continuous benchmarking for model registry
- [ ] Add support for vLLM and Ollama loaders
- [ ] Experiment with context length scaling (32K, 64K, 128K)
- [ ] Fine-tuning experiments on domain-specific tasks

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with [Claude Code](https://claude.com/claude-code) advanced features
- Hardware baseline established 2025-10-02
- Initial embedding benchmark migrated from [Langextract](https://github.com/user/langextract) project

---

**Last Updated**: 2025-10-02
**Claude Code Version**: Latest
**Repository Status**: 🟢 Active Development
