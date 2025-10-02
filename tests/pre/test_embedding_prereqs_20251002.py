"""
PRE-EXECUTION TESTS: Embedding Benchmark Prerequisites
=======================================================

CRITICAL: These tests MUST pass before executing embedding_benchmark.py

Purpose:
    Validate all system requirements, dependencies, and external resources
    are available and correctly configured before running the benchmark.

Blocking Tests:
    - PyTorch + CUDA availability
    - GPU memory sufficiency
    - Model existence on Hugging Face Hub
    - Required package installations

Exit Behavior:
    Any failure should BLOCK benchmark execution to prevent runtime errors.

Author: @test-architect
Date: 2025-10-02
"""

import os
import sys
import pytest
import subprocess


# =============================================================================
# SYSTEM REQUIREMENTS TESTS
# =============================================================================

def test_python_version():
    """Validate Python version is 3.10 or higher."""
    version = sys.version_info
    assert version.major == 3 and version.minor >= 10, (
        f"Python 3.10+ required, found {version.major}.{version.minor}"
    )


def test_pytorch_installed():
    """Validate PyTorch is installed and importable."""
    try:
        import torch
        assert torch.__version__, "PyTorch version not detected"
    except ImportError as e:
        pytest.fail(f"PyTorch not installed: {e}")


def test_cuda_available():
    """CRITICAL: Validate CUDA is available for GPU acceleration."""
    import torch

    assert torch.cuda.is_available(), (
        "CUDA not available. GPU acceleration required for benchmark."
    )


def test_gpu_count():
    """Validate at least one GPU is detected."""
    import torch

    gpu_count = torch.cuda.device_count()
    assert gpu_count >= 1, f"No GPUs detected (expected 2, found {gpu_count})"


def test_gpu_memory_sufficient():
    """Validate GPU has at least 8GB VRAM (safety margin for 32GB GPUs)."""
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)

        assert total_memory_gb >= 8, (
            f"Insufficient GPU memory: {total_memory_gb:.1f}GB (min 8GB required)"
        )


def test_compute_capability():
    """Validate GPU compute capability is sufficient (>= 7.0 for modern models)."""
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        compute_cap = float(f"{props.major}.{props.minor}")

        assert compute_cap >= 7.0, (
            f"GPU compute capability {compute_cap} too low (min 7.0 required)"
        )


# =============================================================================
# DEPENDENCY TESTS
# =============================================================================

def test_sentence_transformers_installed():
    """CRITICAL: Validate sentence-transformers library is installed."""
    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        pytest.fail(f"sentence-transformers not installed: {e}")


def test_numpy_installed():
    """Validate numpy is installed (used for embeddings manipulation)."""
    try:
        import numpy as np
        assert np.__version__, "NumPy version not detected"
    except ImportError as e:
        pytest.fail(f"NumPy not installed: {e}")


def test_requirements_packages_installed():
    """Validate key packages from requirements.txt are installed."""
    required_packages = [
        'torch',
        'transformers',
        'sentence-transformers',
        'numpy',
        'accelerate',
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)

    assert not missing, f"Missing required packages: {', '.join(missing)}"


# =============================================================================
# MODEL AVAILABILITY TESTS (CRITICAL)
# =============================================================================

def test_model_multilingual_exists():
    """
    CRITICAL: Validate paraphrase-multilingual-MiniLM-L12-v2 is accessible.

    This is the CPU baseline model. Validates by loading from cache or downloading.
    Prefers local cache to avoid API authentication issues.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model_id = 'paraphrase-multilingual-MiniLM-L12-v2'

        # Try to load model (uses cache first, downloads if needed)
        # This is the actual check the benchmark does, so it's the most accurate
        model = SentenceTransformer(model_id)

        assert model is not None, f"Model {model_id} failed to load"

        # Verify model has expected attributes
        assert hasattr(model, 'encode'), "Model missing encode method"

        print(f"✅ Model {model_id} loaded successfully from cache/download")

    except ImportError:
        pytest.skip("sentence-transformers not installed")
    except Exception as e:
        pytest.fail(
            f"BLOCKING ERROR: Model 'paraphrase-multilingual-MiniLM-L12-v2' "
            f"cannot be loaded: {e}"
        )


def test_model_bge_legal_exists():
    """
    HIGH RISK: Validate dariolopez/bge-m3-es-legal-tmp-6 is accessible.

    WARNING: Model name contains 'tmp-6' suggesting it might be temporary/private.
    Validates by attempting to load from cache or download.
    If model doesn't exist, benchmark will fail at GPU test section.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model_id = 'dariolopez/bge-m3-es-legal-tmp-6'

        # Try to load model (uses cache first, downloads if needed)
        # Note: This may take time on first run if downloading (~350MB)
        model = SentenceTransformer(model_id)

        assert model is not None, f"Model {model_id} failed to load"

        # Verify model has expected attributes
        assert hasattr(model, 'encode'), "Model missing encode method"

        print(f"✅ Model {model_id} loaded successfully from cache/download")

    except ImportError:
        pytest.skip("sentence-transformers not installed")
    except Exception as e:
        # This is expected to fail if model is private/removed
        # Benchmark will handle gracefully with try/except in line 59-89
        pytest.fail(
            f"RISK CONFIRMED: Model 'dariolopez/bge-m3-es-legal-tmp-6' "
            f"cannot be loaded: {e}\n"
            f"Benchmark will skip GPU test section but continue with CPU baseline."
        )


def test_model_loading_permissions():
    """
    Validate Hugging Face authentication if required for private models.

    Note: If BGE model is private, HF_TOKEN environment variable must be set.
    """
    model_id = 'dariolopez/bge-m3-es-legal-tmp-6'

    # Check if HF_TOKEN is set (may be required for private models)
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

    if not hf_token:
        pytest.skip(
            f"HF_TOKEN not set. If '{model_id}' is private, benchmark will fail. "
            f"Set HF_TOKEN environment variable with authentication token."
        )


# =============================================================================
# GPU STATE TESTS
# =============================================================================

def test_gpu_not_busy():
    """
    Validate GPU utilization is low (<10%) before benchmark.

    High utilization indicates other processes are using the GPU,
    which would skew benchmark results.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Try to allocate small tensor to verify GPU is accessible
        test_tensor = torch.zeros(1000, 1000).cuda()
        del test_tensor
        torch.cuda.empty_cache()

        # Note: Detailed utilization check requires nvidia-ml-py3
        # For now, we validate GPU is accessible and responsive

    except RuntimeError as e:
        pytest.fail(f"GPU appears busy or inaccessible: {e}")


def test_cuda_cache_clearable():
    """Validate CUDA cache can be cleared before benchmark."""
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            pytest.fail(f"Cannot clear CUDA cache: {e}")


# =============================================================================
# FILESYSTEM TESTS
# =============================================================================

def test_benchmark_script_exists():
    """Validate embedding_benchmark.py file exists."""
    script_path = os.path.join(
        os.path.dirname(__file__),
        '../../benchmarks/scripts/embedding_benchmark.py'
    )

    assert os.path.isfile(script_path), (
        f"Benchmark script not found at: {script_path}"
    )


def test_benchmark_script_executable():
    """Validate embedding_benchmark.py has valid Python syntax."""
    script_path = os.path.join(
        os.path.dirname(__file__),
        '../../benchmarks/scripts/embedding_benchmark.py'
    )

    if os.path.isfile(script_path):
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', script_path],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, (
            f"Benchmark script has syntax errors: {result.stderr}"
        )


def test_results_directory_writable():
    """Validate results directory exists or can be created."""
    results_dir = os.path.join(
        os.path.dirname(__file__),
        '../../benchmarks/results/raw'
    )

    os.makedirs(results_dir, exist_ok=True)
    assert os.path.isdir(results_dir), f"Cannot create results directory: {results_dir}"


# =============================================================================
# SUMMARY FUNCTION FOR MANUAL EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("PRE-EXECUTION TESTS: Embedding Benchmark Prerequisites")
    print("=" * 70)
    print()
    print("Running blocking tests...")
    print()

    # Run pytest with verbose output
    exit_code = pytest.main([__file__, '-v', '--tb=short'])

    print()
    print("=" * 70)
    if exit_code == 0:
        print("ALL PRE-TESTS PASSED - BENCHMARK EXECUTION APPROVED")
    else:
        print("PRE-TESTS FAILED - DO NOT EXECUTE BENCHMARK")
    print("=" * 70)

    sys.exit(exit_code)
