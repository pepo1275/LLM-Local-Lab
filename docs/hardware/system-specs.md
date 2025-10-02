# System Hardware Specifications

**Last Updated**: 2025-10-02

## Overview

High-performance workstation optimized for local LLM inference and experimentation.

---

## CPU

**Model**: AMD Ryzen 9 9950X
- **Architecture**: Zen 5
- **Cores**: 16 physical cores
- **Threads**: 32 logical processors
- **Base Clock**: 4.3 GHz
- **Boost Clock**: Up to 5.7 GHz
- **Cache**:
  - L2: 16 MB
  - L3: 64 MB
- **TDP**: 170W

**Performance Characteristics**:
- Excellent for CPU-based inference (fallback when GPU unavailable)
- High core count beneficial for parallel preprocessing
- Strong single-thread performance for sequential tasks

---

## Memory (RAM)

**Capacity**: 126 GB (125.6 GB usable)
- **Type**: DDR5
- **Speed**: High-speed (exact speed TBD)
- **Configuration**: Dual or Quad channel

**LLM Implications**:
- Sufficient for loading large models (70B+ parameters) into system RAM
- Enables offloading when VRAM is insufficient
- Supports massive batch processing for embeddings/preprocessing

---

## Graphics Processing Units (GPUs)

### GPU 0: NVIDIA GeForce RTX 5090

**Specifications**:
- **VRAM**: 32 GB GDDR7
- **Compute Capability**: 12.0 (Blackwell architecture)
- **CUDA Cores**: ~21,760 (estimated)
- **Tensor Cores**: 4th generation (with FP4 support)
- **RT Cores**: 5th generation
- **Memory Bandwidth**: ~1,568 GB/s
- **TDP**: 600W
- **PCIe**: Gen 5 x16

**Idle Stats** (baseline):
- Temperature: 40°C
- Memory Used: 0 MB (idle)
- Utilization: 0%

### GPU 1: NVIDIA GeForce RTX 5090

**Specifications**: Identical to GPU 0

**Idle Stats** (baseline):
- Temperature: 37°C
- Memory Used: 0 MB (idle)
- Utilization: 0%

### Combined GPU Capabilities

**Total VRAM**: 64 GB (2x 32GB)
- Enables running 70B parameter models at Q4 quantization
- Supports dual-model deployment (e.g., LLM + embedding model)
- Massive headroom for context windows and batch processing

**Interconnect**: PCIe Gen 5 (high bandwidth for multi-GPU workloads)

**Thermal Characteristics**:
- Idle: 37-40°C
- Safe operating temperature: <85°C
- Throttling begins: ~83°C

---

## CUDA and Deep Learning Software

**CUDA Toolkit**: 13.0
- Latest generation CUDA support
- Compatible with cutting-edge ML frameworks

**NVIDIA Driver**: 581.42
- Latest stable driver (as of 2025-10-02)
- Full support for RTX 5090 features

**PyTorch**: 2.9.0.dev20250829+cu128
- Development build with CUDA 12.8 support
- Dual-GPU configuration verified and working
- Detects both RTX 5090 GPUs correctly

**Additional Libraries**:
- `torchvision`: 0.24.0.dev20250829+cu128
- `torchaudio`: 2.8.0.dev20250829+cu128

---

## Storage

**Primary Drive (C:)**:
- **Total Capacity**: 3.63 TB (3,999,883,325,440 bytes)
- **Free Space**: 3.39 TB (3,648,175,837,184 bytes)
- **Usage**: ~94% free

**Implications**:
- Ample space for large model files (GGUF, safetensors)
- Room for extensive benchmark result archives
- No storage constraints for foreseeable experiments

---

## Performance Baseline

### Established Benchmarks (2025-10-02)

**Embedding Generation** (sentence-transformers):
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Throughput**: 193.5 documents/second (CPU baseline)
- **GPU Throughput**: 1,326.9 documents/second (GPU 0)
- **Acceleration**: 6.9x faster on GPU vs CPU

**Specialized Legal Model**:
- **Model**: `dariolopez/bge-m3-es-legal-tmp-6`
- **Batch Processing**: 2,398 documents/second (1,000 doc batch)
- **VRAM Usage**: 2.56 GB on GPU 0
- **Free VRAM**: 28.41 GB remaining

**Key Observations**:
- GPU acceleration is significant (6.9x for embeddings)
- Minimal VRAM usage for embedding models (<3GB)
- Massive headroom for simultaneous workloads

---

## Thermal Management

**Cooling Solution**: (To be documented)

**Observed Temperatures**:
- Idle: 37-40°C (both GPUs)
- Under load: (To be measured during LLM benchmarks)

**Recommendations**:
- Monitor temps during long inference runs
- Target: Keep below 80°C for sustained performance
- Consider reducing batch size if temps exceed 85°C

---

## Power Consumption

**Total System TDP** (estimated):
- CPU: 170W (max)
- GPU 0: 600W (max)
- GPU 1: 600W (max)
- Other components: ~150W
- **Total**: ~1,520W under full load

**Power Supply Requirements**: 1600W+ recommended

**Observed Idle Power**: (To be measured with power meter)

---

## Model Capacity Estimates

Based on hardware specifications, estimated maximum model sizes:

### Single GPU (32GB VRAM)

| Model Size | Quantization | Expected VRAM | Fit? |
|------------|--------------|---------------|------|
| 7B | FP16 | ~14 GB | ✅ Yes (comfortable) |
| 13B | FP16 | ~26 GB | ✅ Yes (tight) |
| 30B | Q8 | ~30 GB | ✅ Yes (very tight) |
| 70B | Q4 | ~35 GB | ❌ No (requires dual-GPU) |

### Dual GPU (64GB total VRAM)

| Model Size | Quantization | Expected VRAM | Fit? |
|------------|--------------|---------------|------|
| 70B | Q4 | ~40 GB | ✅ Yes |
| 70B | Q5 | ~48 GB | ✅ Yes |
| 100B | Q4 | ~55 GB | ✅ Yes (tight) |
| 180B | Q4 | ~90 GB | ❌ No (requires offloading) |

**Note**: Estimates include ~15% overhead for activations and KV cache. Actual usage may vary based on context length and batch size.

---

## Upgrade Path

**Current Bottlenecks**: None identified for LLM inference

**Future Considerations**:
- Monitor VRAM usage with 100B+ models (may need CPU offloading)
- Storage is ample, no upgrade needed
- RAM is sufficient for foreseeable workloads

---

## Verification Commands

```bash
# Check CPU info
lscpu  # Linux
wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors  # Windows

# Check RAM
free -h  # Linux
systeminfo | findstr /C:"Total Physical Memory"  # Windows

# Check GPUs
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Check VRAM per GPU
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

---

**Hardware Assessment**: ⭐⭐⭐⭐⭐ (5/5)

This system is **exceptionally well-suited** for local LLM experimentation:
- Dual RTX 5090s provide best-in-class consumer GPU performance
- 64GB total VRAM enables 70B models at Q4/Q5
- High core-count CPU handles preprocessing efficiently
- Abundant RAM allows for CPU offloading if needed
- Storage is not a constraint

**Recommended Use Cases**:
- Production inference for 7B-13B models (FP16, maximum speed)
- Experimentation with 70B models (Q4/Q5, dual-GPU)
- Embedding generation at scale (thousands of docs/second)
- Multi-model deployments (LLM + embedding model simultaneously)
- Context length experiments (large KV cache supported)
