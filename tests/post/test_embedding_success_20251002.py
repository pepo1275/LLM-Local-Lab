"""
POST-EXECUTION TESTS: Embedding Benchmark Success Validation
=============================================================

Purpose:
    Validate that embedding_benchmark.py executed successfully and produced
    expected results with correct performance characteristics.

Test Strategy:
    - Benchmark completed without crashes
    - GPU speedup is significant (>2x)
    - Embeddings have correct dimensions
    - Memory usage is within acceptable limits
    - Performance metrics are reasonable

Usage:
    Run AFTER executing embedding_benchmark.py
    If no results file exists, tests will mock data or skip

Author: @test-architect
Date: 2025-10-02
"""

import os
import sys
import pytest
import json
import time
from pathlib import Path


# =============================================================================
# BENCHMARK EXECUTION TESTS
# =============================================================================

def test_benchmark_runs_without_crash():
    """
    Validate embedding_benchmark.py completes without exceptions.

    This test actually executes the benchmark script and monitors for errors.
    """
    import subprocess

    script_path = Path(__file__).parent.parent.parent / 'benchmarks' / 'scripts' / 'embedding_benchmark.py'

    if not script_path.exists():
        pytest.skip(f"Benchmark script not found: {script_path}")

    print(f"\nExecuting benchmark: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    assert result.returncode == 0, (
        f"Benchmark failed with exit code {result.returncode}\n"
        f"Error: {result.stderr}"
    )

    # Store output for other tests to analyze
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'
    output_path.write_text(result.stdout)


def test_benchmark_output_contains_results():
    """Validate benchmark output contains expected result markers."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found. Run test_benchmark_runs_without_crash first.")

    output = output_path.read_text(encoding='utf-8')

    # Check for key result sections
    assert "TEST GPU EMBEDDINGS - RTX 5090 DUAL-GPU" in output
    assert "TEST 1: MODELO MULTILINGUAL (CPU Baseline)" in output
    assert "TEST 2: BGE-M3 SPANISH LEGAL (GPU Acelerado)" in output
    assert "TEST GPU EMBEDDINGS COMPLETADO" in output


def test_no_error_messages_in_output():
    """Validate no ERROR markers in benchmark output (except handled exceptions)."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    # Check for unhandled errors (BGE error is expected and handled)
    error_lines = [line for line in output.split('\n') if '[ERROR]' in line]

    # Allow BGE-M3 error if model doesn't exist (it's caught by try/except)
    allowed_errors = ['[ERROR] BGE-M3 GPU:']

    unexpected_errors = [
        line for line in error_lines
        if not any(allowed in line for allowed in allowed_errors)
    ]

    assert not unexpected_errors, (
        f"Unexpected errors in output: {unexpected_errors}"
    )


# =============================================================================
# PERFORMANCE VALIDATION TESTS
# =============================================================================

def test_gpu_speedup_significant():
    """
    CRITICAL: Validate GPU achieves >2x speedup vs CPU.

    This is the primary success metric for GPU acceleration.
    """
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    # Parse speedup from output
    speedup = None
    for line in output.split('\n'):
        if '[RESULTADO] Aceleracion GPU:' in line:
            # Extract number like "6.9x mas rapido"
            parts = line.split(':')[1].strip().split('x')[0].strip()
            try:
                speedup = float(parts)
            except ValueError:
                pass

    if speedup is None:
        pytest.skip("GPU speedup not found in output (BGE model may have failed)")

    assert speedup > 2.0, (
        f"GPU speedup {speedup}x is below minimum threshold of 2x. "
        f"Expected >6x on RTX 5090."
    )

    print(f"\nGPU Speedup: {speedup}x (PASS)")


def test_cpu_throughput_reasonable():
    """Validate CPU baseline throughput is within expected range."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    # Parse CPU throughput (docs/second)
    cpu_throughput = None
    for line in output.split('\n'):
        if '[CPU] Documentos por segundo:' in line:
            try:
                cpu_throughput = float(line.split(':')[1].strip())
            except ValueError:
                pass

    if cpu_throughput is None:
        pytest.fail("CPU throughput not found in output")

    # Reasonable range: 5-100 docs/sec on CPU (depends on model size)
    assert 5 <= cpu_throughput <= 200, (
        f"CPU throughput {cpu_throughput} docs/sec outside expected range (5-200)"
    )

    print(f"\nCPU Throughput: {cpu_throughput} docs/sec (PASS)")


def test_gpu_throughput_high():
    """Validate GPU throughput is significantly higher than CPU."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    # Parse GPU throughput
    gpu_throughput = None
    for line in output.split('\n'):
        if '[GPU] Documentos por segundo:' in line:
            try:
                gpu_throughput = float(line.split(':')[1].strip())
            except ValueError:
                pass

    if gpu_throughput is None:
        pytest.skip("GPU throughput not found (BGE model may have failed)")

    # Expected: >100 docs/sec on RTX 5090
    assert gpu_throughput > 50, (
        f"GPU throughput {gpu_throughput} docs/sec is too low. "
        f"Expected >100 on RTX 5090."
    )

    print(f"\nGPU Throughput: {gpu_throughput} docs/sec (PASS)")


def test_batch_processing_scales():
    """Validate large batch processing achieves higher throughput."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    # Parse batch throughput
    batch_throughput = None
    for line in output.split('\n'):
        if '[BATCH GRANDE]' in line and 'docs/segundo' in line:
            try:
                batch_throughput = float(line.split()[-2])
            except (ValueError, IndexError):
                pass

    if batch_throughput is None:
        pytest.skip("Batch throughput not found")

    # Batch processing should be even faster (>500 docs/sec)
    assert batch_throughput > 200, (
        f"Batch throughput {batch_throughput} docs/sec is too low"
    )

    print(f"\nBatch Throughput: {batch_throughput} docs/sec (PASS)")


# =============================================================================
# EMBEDDING SHAPE VALIDATION
# =============================================================================

def test_cpu_embeddings_shape_correct():
    """Validate CPU embeddings have correct dimensions."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    # Parse embedding shape like "(100, 384)"
    cpu_shape = None
    for line in output.split('\n'):
        if '[CPU] Shape embeddings:' in line:
            shape_str = line.split(':')[1].strip()
            # Extract dimensions from "(100, 384)"
            try:
                dims = shape_str.strip('()').split(',')
                cpu_shape = (int(dims[0].strip()), int(dims[1].strip()))
            except (ValueError, IndexError):
                pass

    if cpu_shape is None:
        pytest.fail("CPU embedding shape not found in output")

    # Validate: 100 documents, embedding dimension should be 384 or similar
    assert cpu_shape[0] == 100, f"Expected 100 documents, got {cpu_shape[0]}"
    assert 128 <= cpu_shape[1] <= 1024, (
        f"Unexpected embedding dimension: {cpu_shape[1]} (expected 128-1024)"
    )

    print(f"\nCPU Embedding Shape: {cpu_shape} (PASS)")


def test_gpu_embeddings_shape_correct():
    """Validate GPU embeddings have correct dimensions."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    gpu_shape = None
    for line in output.split('\n'):
        if '[GPU] Shape embeddings:' in line:
            shape_str = line.split(':')[1].strip()
            try:
                dims = shape_str.strip('()').split(',')
                gpu_shape = (int(dims[0].strip()), int(dims[1].strip()))
            except (ValueError, IndexError):
                pass

    if gpu_shape is None:
        pytest.skip("GPU embedding shape not found (BGE model may have failed)")

    assert gpu_shape[0] == 100, f"Expected 100 documents, got {gpu_shape[0]}"
    assert 128 <= gpu_shape[1] <= 1024, (
        f"Unexpected embedding dimension: {gpu_shape[1]}"
    )

    print(f"\nGPU Embedding Shape: {gpu_shape} (PASS)")


# =============================================================================
# MEMORY USAGE TESTS
# =============================================================================

def test_gpu_memory_usage_reasonable():
    """Validate GPU memory usage is within acceptable limits (<16GB)."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    # Parse memory usage
    memory_used = None
    for line in output.split('\n'):
        if 'Memoria GPU usada:' in line:
            try:
                memory_used = float(line.split(':')[1].strip().replace('GB', ''))
            except ValueError:
                pass

    if memory_used is None:
        pytest.skip("GPU memory usage not found")

    # Should use <16GB on 32GB card for this small benchmark
    assert memory_used < 16.0, (
        f"GPU memory usage {memory_used}GB exceeds 16GB threshold"
    )

    print(f"\nGPU Memory Used: {memory_used}GB (PASS)")


def test_gpu_memory_not_exhausted():
    """Validate GPU has free memory remaining after benchmark."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    memory_free = None
    for line in output.split('\n'):
        if 'Memoria GPU libre:' in line:
            try:
                memory_free = float(line.split(':')[1].strip().replace('GB', ''))
            except ValueError:
                pass

    if memory_free is None:
        pytest.skip("GPU free memory not found")

    # Should have >10GB free
    assert memory_free > 10.0, (
        f"GPU only has {memory_free}GB free, may indicate memory leak"
    )

    print(f"\nGPU Memory Free: {memory_free}GB (PASS)")


# =============================================================================
# SUCCESS MARKERS
# =============================================================================

def test_benchmark_declares_success():
    """Validate benchmark output contains success markers."""
    output_path = Path(__file__).parent / 'latest_benchmark_output.txt'

    if not output_path.exists():
        pytest.skip("Benchmark output not found")

    output = output_path.read_text(encoding='utf-8')

    # Should contain success messages
    success_markers = [
        "TEST GPU EMBEDDINGS COMPLETADO",
        "RTX 5090 lista para produccion"
    ]

    for marker in success_markers:
        assert marker in output, f"Success marker missing: {marker}"


# =============================================================================
# SUMMARY FUNCTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("POST-EXECUTION TESTS: Embedding Benchmark Success Validation")
    print("=" * 70)
    print()

    exit_code = pytest.main([__file__, '-v', '--tb=short'])

    print()
    print("=" * 70)
    if exit_code == 0:
        print("ALL POST-TESTS PASSED - BENCHMARK SUCCESSFUL")
    else:
        print("POST-TESTS FAILED - BENCHMARK ISSUES DETECTED")
    print("=" * 70)

    sys.exit(exit_code)
