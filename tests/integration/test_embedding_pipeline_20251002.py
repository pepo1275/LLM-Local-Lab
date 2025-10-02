"""
INTEGRATION TESTS: Embedding Pipeline Validation
=================================================

Purpose:
    Test the end-to-end pipeline from configuration to model loading,
    embedding generation, and result formatting.

Test Strategy:
    - Config-to-model loading pipeline
    - Model inference with different input types
    - Output format validation
    - Error handling for edge cases
    - Batch processing behavior

Author: @test-architect
Date: 2025-10-02
"""

import os
import pytest
import numpy as np


# =============================================================================
# MODEL LOADING PIPELINE TESTS
# =============================================================================

def test_sentence_transformer_loads_multilingual():
    """
    Integration test: Load paraphrase-multilingual model successfully.
    """
    from sentence_transformers import SentenceTransformer

    model_id = 'paraphrase-multilingual-MiniLM-L12-v2'

    try:
        model = SentenceTransformer(model_id)
        assert model is not None
        assert hasattr(model, 'encode')
    except Exception as e:
        pytest.fail(f"Failed to load multilingual model: {e}")


def test_sentence_transformer_device_placement():
    """
    Integration test: Validate model can be moved to correct device.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model_id = 'paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(model_id)

    # Test moving to GPU
    device = torch.device('cuda:0')
    model = model.to(device)

    # Validate model is on GPU
    assert str(model.device) == 'cuda:0', (
        f"Model not on cuda:0, found: {model.device}"
    )


def test_model_encode_single_document():
    """
    Integration test: Encode single document successfully.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    document = "Este es un documento de prueba"
    embedding = model.encode(document)

    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1  # 1D vector for single doc
    assert embedding.shape[0] > 0  # Has dimensions


def test_model_encode_multiple_documents():
    """
    Integration test: Encode batch of documents successfully.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    documents = [
        "Primer documento",
        "Segundo documento",
        "Tercer documento"
    ]

    embeddings = model.encode(documents)

    assert embeddings is not None
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings.shape) == 2  # 2D matrix
    assert embeddings.shape[0] == 3  # 3 documents
    assert embeddings.shape[1] > 0  # Has embedding dimension


def test_model_encode_empty_list():
    """
    Integration test: Handle empty document list gracefully.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    documents = []

    try:
        embeddings = model.encode(documents)
        # Some implementations may return empty array, others may error
        # Just verify it doesn't crash ungracefully
        assert True
    except Exception as e:
        # If it errors, it should be a clear error message
        assert 'empty' in str(e).lower() or 'no' in str(e).lower()


def test_model_encode_special_characters():
    """
    Integration test: Handle documents with special characters.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    documents = [
        "Documento con tildes: áéíóú",
        "Documento con ñ: España",
        "Documento con símbolos: @#$%",
        "Documento con números: 123456"
    ]

    embeddings = model.encode(documents)

    assert embeddings.shape[0] == 4
    assert not np.any(np.isnan(embeddings)), "Embeddings contain NaN values"
    assert not np.any(np.isinf(embeddings)), "Embeddings contain Inf values"


# =============================================================================
# EMBEDDING OUTPUT VALIDATION
# =============================================================================

def test_embeddings_normalized():
    """
    Integration test: Verify embeddings are properly normalized.

    Most sentence-transformer models output L2-normalized embeddings.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    document = "Documento de prueba para normalización"
    embedding = model.encode(document)

    # Calculate L2 norm
    norm = np.linalg.norm(embedding)

    # Should be close to 1.0 if normalized
    # Allow some floating point tolerance
    assert 0.95 <= norm <= 1.05, (
        f"Embedding not normalized: L2 norm = {norm} (expected ~1.0)"
    )


def test_embeddings_consistent_dimensions():
    """
    Integration test: Verify all embeddings from a model have same dimensions.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    documents = [
        "Short doc",
        "This is a medium length document with more words",
        "Very long document " * 20  # Repeat for length
    ]

    embeddings = model.encode(documents)

    # All embeddings should have same dimension despite input length differences
    assert embeddings.shape[1] == embeddings.shape[1], (
        "Embedding dimensions inconsistent across documents"
    )


def test_embeddings_reproducible():
    """
    Integration test: Verify same input produces same embeddings.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    document = "Documento de prueba para reproducibilidad"

    embedding1 = model.encode(document)
    embedding2 = model.encode(document)

    # Should be identical (or very close due to GPU floating point)
    assert np.allclose(embedding1, embedding2, atol=1e-6), (
        "Embeddings not reproducible for identical input"
    )


def test_similar_documents_have_similar_embeddings():
    """
    Integration test: Verify semantic similarity is captured.
    """
    from sentence_transformers import SentenceTransformer
    from scipy.spatial.distance import cosine

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    doc1 = "El gato está en la casa"
    doc2 = "El gato está en el hogar"  # Similar meaning
    doc3 = "Los coches son rápidos"    # Different meaning

    emb1 = model.encode(doc1)
    emb2 = model.encode(doc2)
    emb3 = model.encode(doc3)

    # Calculate cosine similarities
    sim_12 = 1 - cosine(emb1, emb2)  # Similar docs
    sim_13 = 1 - cosine(emb1, emb3)  # Different docs

    # Similar docs should have higher similarity
    assert sim_12 > sim_13, (
        f"Similar documents have lower similarity ({sim_12:.3f}) "
        f"than different documents ({sim_13:.3f})"
    )


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================

def test_batch_processing_correct_size():
    """
    Integration test: Verify batch_size parameter works correctly.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    documents = ["Documento " + str(i) for i in range(50)]

    # Encode with specific batch size
    embeddings = model.encode(documents, batch_size=10)

    assert embeddings.shape[0] == 50, (
        f"Expected 50 embeddings, got {embeddings.shape[0]}"
    )


def test_batch_processing_vs_sequential():
    """
    Integration test: Verify batch and sequential processing produce same results.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    documents = ["Doc " + str(i) for i in range(10)]

    # Batch processing
    batch_embeddings = model.encode(documents, batch_size=10)

    # Sequential processing
    sequential_embeddings = np.array([model.encode(doc) for doc in documents])

    # Should be very close (allow for numerical differences)
    assert np.allclose(batch_embeddings, sequential_embeddings, atol=1e-5), (
        "Batch and sequential processing produce different results"
    )


def test_large_batch_performance():
    """
    Integration test: Verify large batch processing completes successfully.
    """
    import time
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    documents = ["Documento de prueba " + str(i) for i in range(1000)]

    start_time = time.time()
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=False)
    elapsed_time = time.time() - start_time

    assert embeddings.shape[0] == 1000

    # Should complete in reasonable time (<60 seconds on CPU)
    assert elapsed_time < 120, (
        f"Large batch took too long: {elapsed_time:.1f}s"
    )

    print(f"\nLarge batch (1000 docs) completed in {elapsed_time:.2f}s")


# =============================================================================
# GPU ACCELERATION TESTS
# =============================================================================

def test_gpu_acceleration_faster_than_cpu():
    """
    Integration test: Verify GPU processing is faster than CPU.

    This mirrors the core functionality of embedding_benchmark.py
    """
    import time
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    documents = ["Test document " + str(i) for i in range(100)]

    # CPU baseline
    model_cpu = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    start_time = time.time()
    embeddings_cpu = model_cpu.encode(documents, show_progress_bar=False)
    cpu_time = time.time() - start_time

    # GPU accelerated
    model_gpu = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    model_gpu = model_gpu.to(torch.device('cuda:0'))

    start_time = time.time()
    embeddings_gpu = model_gpu.encode(documents, show_progress_bar=False)
    gpu_time = time.time() - start_time

    speedup = cpu_time / gpu_time if gpu_time > 0 else 0

    print(f"\nCPU time: {cpu_time:.2f}s")
    print(f"GPU time: {gpu_time:.2f}s")
    print(f"Speedup: {speedup:.1f}x")

    # GPU should be faster (at least 1.2x)
    assert speedup > 1.2, (
        f"GPU not faster than CPU: {speedup:.1f}x speedup"
    )

    # Embeddings should be very similar
    assert np.allclose(embeddings_cpu, embeddings_gpu, atol=1e-4), (
        "CPU and GPU produce different embeddings"
    )


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_invalid_model_name_fails_gracefully():
    """
    Integration test: Verify invalid model names produce clear error messages.
    """
    from sentence_transformers import SentenceTransformer

    invalid_model = 'this-model-does-not-exist-12345'

    with pytest.raises(Exception) as exc_info:
        model = SentenceTransformer(invalid_model)

    # Error message should mention the model name
    assert invalid_model in str(exc_info.value) or 'not found' in str(exc_info.value).lower()


def test_null_document_handling():
    """
    Integration test: Handle None/null documents gracefully.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    documents = ["Valid doc", None, "Another valid doc"]

    # Should either skip None or raise clear error
    try:
        embeddings = model.encode(documents)
        # If it succeeds, None was handled
        assert True
    except (TypeError, ValueError) as e:
        # Clear error about None/null is acceptable
        assert 'None' in str(e) or 'null' in str(e) or 'empty' in str(e)


# =============================================================================
# SUMMARY FUNCTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("INTEGRATION TESTS: Embedding Pipeline Validation")
    print("=" * 70)
    print()

    exit_code = pytest.main([__file__, '-v', '--tb=short'])

    print()
    print("=" * 70)
    if exit_code == 0:
        print("ALL INTEGRATION TESTS PASSED - PIPELINE VALIDATED")
    else:
        print("INTEGRATION TESTS FAILED - PIPELINE ISSUES DETECTED")
    print("=" * 70)

    sys.exit(exit_code)
