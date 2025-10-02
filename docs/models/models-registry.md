# Models Registry

Comprehensive registry of all LLM models tested on this system.

**Last Updated**: 2025-10-02

---

## Registry Format

Each entry includes:
- Model name and variant
- Test date and hardware used
- Quantization type
- Performance metrics (throughput, latency, VRAM)
- Quality observations
- Recommendation rating (1-5 stars)
- Links to configs and benchmark reports

---

## Tested Models

### Embedding Models

#### paraphrase-multilingual-MiniLM-L12-v2

**Tested**: 2025-10-02
**Hardware**: Dual RTX 5090 (used GPU 0 only)
**Source**: sentence-transformers

**Configuration**:
- Precision: FP16 (auto)
- Device: CUDA:0
- Batch size: Default

**Performance** (2025-10-02 Re-test):
- **GPU Throughput**: 553.3 docs/second (100 docs batch)
- **Latency**: 2.0ms per document
- **Total Time**: 0.18s (100 docs)
- **VRAM Usage**: ~1 GB (estimated)
- **Device**: cuda:0 (⚠️ Note: Listed as "CPU baseline" but ran on GPU)

**Quality Notes**:
- Multilingual support (384-dim embeddings)
- Good for general-purpose semantic search
- Lightweight and fast

**Use Cases**:
- Document embedding generation
- Semantic search
- Clustering and classification

**Recommendation**: ⭐⭐⭐⭐ (4/5) - Excellent general-purpose embedding model

---

#### dariolopez/bge-m3-es-legal-tmp-6

**Tested**: 2025-10-02
**Hardware**: Dual RTX 5090 (used GPU 0 only)
**Source**: HuggingFace

**Configuration**:
- Precision: FP16
- Device: CUDA:0
- Batch size: 32 (for large batch test)

**Performance** (2025-10-02 Re-test):
- **Small Batch** (100 docs): 1,749.2 docs/second
- **Large Batch** (1,000 docs): 2,422.9 docs/second
- **Latency**: 1.0ms per doc (small), 0.41ms (large)
- **Total Time**: 0.06s (100 docs), 0.41s (1000 docs)
- **VRAM Usage**: 2.56 GB (peak)
- **VRAM Reserved**: 2.59 GB
- **Free VRAM**: 28.41 GB remaining (88% unused)
- **Batch Scaling**: 1.39x (⚠️ Suboptimal - needs larger batches)

**Quality Notes**:
- Specialized for Spanish legal/administrative documents
- 1,024-dimensional embeddings (higher than MiniLM)
- Excellent domain-specific performance

**Use Cases**:
- Spanish legal document processing
- Municipal/administrative text analysis
- Domain-specific semantic search

**Recommendation**: ⭐⭐⭐⭐⭐ (5/5) - Best-in-class for Spanish legal domain

**Benchmark Report**: [embedding_benchmark_20251002.md](../benchmarks/results/reports/embedding_benchmark_20251002.md) (to be generated)

---

## Models Planned for Testing

### LLM Models

#### Llama 3.1 8B

**Status**: 📝 Planned
**Estimated VRAM**: 16 GB (FP16) or 8 GB (Q8)
**Expected Performance**: High (single GPU, full speed)
**Target Use Case**: Fast general-purpose inference

---

#### Llama 3.1 70B Q4

**Status**: 📝 Planned - High Priority
**Estimated VRAM**: 40-45 GB (requires dual-GPU)
**Expected Performance**: Medium (dual-GPU overhead)
**Target Use Case**: High-quality reasoning, long-form generation

**Configuration Strategy**:
- Tensor parallelism across 2x RTX 5090
- Q4_K_M quantization
- Flash Attention 2 enabled
- Context: 8K initially, test up to 32K

---

#### Mixtral 8x7B

**Status**: 📝 Planned
**Estimated VRAM**: 28-32 GB (Q5) single GPU
**Expected Performance**: High
**Target Use Case**: Mixture-of-experts architecture testing

---

#### Qwen 2.5 72B Q4

**Status**: 📝 Planned
**Estimated VRAM**: 42-48 GB (dual-GPU)
**Expected Performance**: Medium-High
**Target Use Case**: Multilingual, coding tasks

---

## Model Categories

### By Size Class

**Small (1B-7B)**:
- Fast inference (>100 tokens/s expected)
- Single GPU, comfortable VRAM
- Suitable for: Real-time applications, simple tasks

**Medium (13B-34B)**:
- Balanced performance (50-100 tokens/s)
- Single GPU at Q4/Q5, or dual-GPU at FP16
- Suitable for: General-purpose applications

**Large (70B+)**:
- Slower inference (20-50 tokens/s)
- Dual-GPU required at Q4/Q5
- Suitable for: Complex reasoning, research

### By Quantization

**FP16** (Full Precision):
- Best quality
- 2 bytes/parameter
- Use when: VRAM allows, quality critical

**Q8** (8-bit Quantization):
- Minimal quality loss
- 1 byte/parameter
- Use when: Balance of quality and size

**Q5** (5-bit Quantization):
- Slight quality loss
- 0.625 bytes/parameter
- Use when: Need to fit larger models

**Q4** (4-bit Quantization):
- Noticeable but acceptable quality loss
- 0.5 bytes/parameter
- Use when: Maximizing model size on available VRAM

**Q3/Q2** (Extreme Quantization):
- Significant quality degradation
- 0.375 / 0.25 bytes/parameter
- Use when: Experimental only, last resort

---

## Benchmarking Methodology

All models are tested with:

1. **Hardware**: AMD Ryzen 9 9950X + 2x RTX 5090 (64GB VRAM)
2. **Software**: PyTorch 2.9.0.dev+cu128, CUDA 13.0
3. **Metrics Captured**:
   - Throughput (tokens/second or docs/second)
   - Time to First Token (TTFT)
   - Total inference time
   - VRAM usage (peak and average)
   - GPU utilization per device
   - Temperature and power draw

4. **Test Dataset**: Standardized prompts (TBD - create in benchmarks/configs/)
5. **Runs**: 3 warmup + 10 test runs (average reported)

---

## Performance Comparison

### Embedding Models (docs/second)

| Model | Dimensions | Batch 100 | Batch 1000 | VRAM |
|-------|-----------|-----------|------------|------|
| MiniLM-L12-v2 | 384 | 1,327 | N/A | <1 GB |
| BGE-M3-ES-Legal | 1,024 | 1,327 | 2,399 | 2.56 GB |

### LLM Models (tokens/second)

| Model | Size | Quant | Throughput | VRAM | Status |
|-------|------|-------|------------|------|--------|
| TBD | - | - | - | - | 📝 Planned |

---

## Recommendations by Use Case

### Real-Time Chat Applications
- **Best**: Llama 3.1 8B Q8 (fast, quality)
- **Alternative**: Mixtral 8x7B Q5 (expertise)

### Research and Reasoning
- **Best**: Llama 3.1 70B Q4 (highest capability on hardware)
- **Alternative**: Qwen 2.5 72B Q4 (multilingual)

### Document Embeddings
- **General Purpose**: MiniLM-L12-v2 (fast, multilingual)
- **Spanish Legal**: BGE-M3-ES-Legal (domain-specific)

### Context-Heavy Tasks
- **Best**: Models with long context support (TBD - test Llama 3.1 at 128K)
- **Strategy**: Lower quantization if VRAM allows

---

## Notes

- All benchmarks reproducible via configs in `models/configs/`
- VRAM estimates include 15-20% overhead for activations
- Actual performance varies with context length and batch size
- Temperature monitoring critical for sustained workloads

---

**Next Steps**:
1. [ ] Create standardized test prompts dataset
2. [ ] Benchmark Llama 3.1 8B (baseline LLM)
3. [ ] Benchmark Llama 3.1 70B Q4 (dual-GPU flagship)
4. [ ] Compare quantization levels (Q4 vs Q5 vs Q8)
5. [ ] Document context length scaling experiments
