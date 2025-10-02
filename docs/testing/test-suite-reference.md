# Test Suite Reference - LLM-Local-Lab

**Última actualización:** 2025-10-02
**Metodología:** RPVEA-A (Agent-Augmented Testing-First)
**Generado por:** @test-architect agent

---

## 🎯 Resumen Ejecutivo

Este documento centraliza toda la información sobre los tests del proyecto, facilitando:
- ✅ Rápida consulta de qué tests existen
- ✅ Cómo ejecutarlos sin investigar cada vez
- ✅ Qué valida cada test suite
- ✅ Cuándo usar cada tipo de test

**Total de Test Suites:** 3 (PRE, POST, Integration)
**Total de Test Functions:** 45
**Cobertura:** Embedding benchmarks (primera implementación)

---

## 📁 Estructura de Testing

```
tests/
├── pre/                              # PRE-tests (baseline validation)
│   └── test_embedding_prereqs_20251002.py
├── post/                             # POST-tests (acceptance criteria)
│   └── test_embedding_success_20251002.py
├── integration/                      # Integration tests (pipeline validation)
│   └── test_embedding_pipeline_20251002.py
└── __init__.py                       # Test discovery
```

---

## 🔴 PRE-TESTS: Baseline Validation

### Archivo: `tests/pre/test_embedding_prereqs_20251002.py`

**Propósito:** Validar que el sistema está listo ANTES de ejecutar benchmarks

**Cuándo ejecutar:**
- ✅ SIEMPRE antes de ejecutar embedding_benchmark.py
- ✅ Parte de VALIDATE phase en RPVEA-A
- ✅ Después de cambios en environment (actualizaciones de libs, etc.)

**Ejecución:**
```bash
# Opción 1: Ejecutar como script
python tests/pre/test_embedding_prereqs_20251002.py

# Opción 2: Con pytest (más detallado)
pytest tests/pre/test_embedding_prereqs_20251002.py -v

# Opción 3: Solo tests críticos (rápido)
pytest tests/pre/test_embedding_prereqs_20251002.py -k "cuda or gpu" -v
```

**Exit Codes:**
- `0` = ✅ Todos los tests pasaron - SAFE TO PROCEED
- `>0` = ❌ Al menos un test falló - DO NOT EXECUTE

---

### Tests Incluidos (17 total)

#### **Categoría 1: Python Environment (2 tests)**

##### `test_python_version()`
```python
def test_python_version():
    """Python 3.10+ required for modern typing and async features"""
```
- **Valida:** Python >= 3.10
- **Crítico:** ⚠️ Medium (features modernas)
- **Si falla:** Actualizar Python

##### `test_required_packages_installed()`
```python
def test_required_packages_installed():
    """Core dependencies must be importable"""
```
- **Valida:** torch, sentence_transformers, numpy, transformers
- **Crítico:** 🔴 HIGH (bloqueante)
- **Si falla:** `pip install -r requirements.txt`

---

#### **Categoría 2: GPU & CUDA (5 tests)**

##### `test_cuda_available()`
```python
def test_cuda_available():
    """CUDA must be available for GPU acceleration"""
```
- **Valida:** `torch.cuda.is_available() == True`
- **Crítico:** 🔴 HIGH (GPU required)
- **Si falla:** Verificar drivers NVIDIA, reinstalar PyTorch con CUDA

##### `test_gpu_count()`
```python
def test_gpu_count():
    """At least one GPU required"""
```
- **Valida:** `torch.cuda.device_count() >= 1`
- **Crítico:** 🔴 HIGH
- **Si falla:** Verificar GPUs detectadas con `nvidia-smi`

##### `test_gpu_memory_sufficient()`
```python
def test_gpu_memory_sufficient():
    """GPU must have at least 8GB VRAM"""
```
- **Valida:** GPU 0 tiene >= 8GB VRAM
- **Crítico:** ⚠️ Medium (embedding models necesitan ~2-4GB)
- **Si falla:** Liberar VRAM o usar GPU más grande

##### `test_gpu_idle()`
```python
def test_gpu_idle():
    """GPU should not be under heavy load"""
```
- **Valida:** Utilización < 50%
- **Crítico:** 🟡 LOW (recomendado para benchmarks precisos)
- **Si falla:** Cerrar procesos que usan GPU

##### `test_gpu_compute_capability()`
```python
def test_gpu_compute_capability():
    """Compute capability 7.0+ recommended"""
```
- **Valida:** Compute capability >= 7.0 (Volta+)
- **Crítico:** 🟡 LOW (RTX 5090 tiene 12.0)
- **Si falla:** GPU muy antigua (no bloqueante)

---

#### **Categoría 3: Model Availability (3 tests)**

##### `test_sentence_transformers_installed()`
```python
def test_sentence_transformers_installed():
    """sentence-transformers package required"""
```
- **Valida:** Puede importar `SentenceTransformer`
- **Crítico:** 🔴 HIGH
- **Si falla:** `pip install sentence-transformers`

##### `test_model_multilingual_exists()` ⚠️ **CONOCIDO: Puede Fallar**
```python
def test_model_multilingual_exists():
    """Validate paraphrase-multilingual model accessible"""
```
- **Valida:** Modelo en Hugging Face Hub (API check)
- **Crítico:** ⚠️ Medium (PERO: modelo en caché local funciona)
- **Si falla:**
  - Opción A: Ignorar si modelo está en caché local
  - Opción B: Autenticar con `huggingface-cli login`
- **Caché local:** `C:\Users\Gamer\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`

##### `test_model_bge_legal_exists()` ⚠️ **SKIPPED si no hay HF_TOKEN**
```python
@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not set")
def test_model_bge_legal_exists():
    """Validate BGE legal model accessible"""
```
- **Valida:** Modelo BGE legal en HF Hub
- **Crítico:** 🟡 LOW (opcional, modelo GPU)
- **Si falla:** Modelo temporal puede no existir
- **Caché local:** `C:\Users\Gamer\.cache\huggingface\hub\models--dariolopez--bge-m3-es-legal-tmp-6`

---

#### **Categoría 4: Dependencies (4 tests)**

##### `test_torch_version()`
```python
def test_torch_version():
    """PyTorch 2.0+ required"""
```
- **Valida:** PyTorch >= 2.0
- **Crítico:** ⚠️ Medium
- **Estado actual:** ✅ 2.9.0.dev (OK)

##### `test_numpy_available()`
```python
def test_numpy_available():
    """NumPy required for array operations"""
```
- **Valida:** NumPy importable
- **Crítico:** 🔴 HIGH

##### `test_transformers_available()`
```python
def test_transformers_available():
    """transformers library for model backends"""
```
- **Valida:** `transformers` importable
- **Crítico:** 🔴 HIGH

##### `test_accelerate_available()`
```python
def test_accelerate_available():
    """accelerate library for GPU optimizations"""
```
- **Valida:** `accelerate` importable
- **Crítico:** ⚠️ Medium (ayuda con multi-GPU)

---

#### **Categoría 5: File System (3 tests)**

##### `test_benchmark_script_exists()`
```python
def test_benchmark_script_exists():
    """Embedding benchmark script must exist"""
```
- **Valida:** `benchmarks/scripts/embedding_benchmark.py` existe
- **Crítico:** 🔴 HIGH
- **Ruta:** `C:\Users\Gamer\Dev\LLM-Local-Lab\benchmarks\scripts\embedding_benchmark.py`

##### `test_results_directory_writable()`
```python
def test_results_directory_writable():
    """Results directory must be writable"""
```
- **Valida:** Puede escribir en `benchmarks/results/raw/`
- **Crítico:** ⚠️ Medium (necesario para guardar outputs)

##### `test_disk_space_sufficient()`
```python
def test_disk_space_sufficient():
    """At least 10GB free space required"""
```
- **Valida:** >= 10GB libres en disco
- **Crítico:** 🟡 LOW (3.63 TB disponibles)

---

### Resultados del Último Run (2025-10-02)

```
✅ PASSED: 15/17 tests
❌ FAILED: 1 test (test_model_multilingual_exists - API auth)
⏩ SKIPPED: 1 test (test_model_bge_legal_exists - no HF_TOKEN)
```

**Análisis:**
- ✅ Environment completamente funcional
- ✅ GPUs idle y listas
- ✅ Modelos en caché local (API check falló pero caché OK)
- ⚠️ HF API requiere auth (no bloqueante si modelos en caché)

---

## 🟢 POST-TESTS: Acceptance Criteria

### Archivo: `tests/post/test_embedding_success_20251002.py`

**Propósito:** Validar que el benchmark ejecutó correctamente y cumple criterios de aceptación

**Cuándo ejecutar:**
- ✅ DESPUÉS de ejecutar embedding_benchmark.py
- ✅ Parte de ASSESS phase en RPVEA-A
- ✅ Para validar optimizaciones (cambios en config, código)

**Ejecución:**
```bash
# Opción 1: Todos los tests
python tests/post/test_embedding_success_20251002.py

# Opción 2: Con pytest (detallado)
pytest tests/post/test_embedding_success_20251002.py -v -s

# Opción 3: Solo tests de performance
pytest tests/post/test_embedding_success_20251002.py -k "speedup or throughput" -v
```

---

### Tests Incluidos (12 total)

#### **Categoría 1: Execution Validation (2 tests)**

##### `test_benchmark_runs_without_crash()` ⚠️ **EJECUTA EL BENCHMARK**
```python
def test_benchmark_runs_without_crash():
    """Benchmark must complete without errors"""
    # ⚠️ IMPORTANTE: Este test EJECUTA el benchmark!
```
- **Acción:** Ejecuta `embedding_benchmark.py` y captura output
- **Valida:** Exit code == 0 (sin crashes)
- **Duración:** 15-20 minutos (incluye descarga de modelos si no existen)
- **Output:** Guardado en variable global `BENCHMARK_OUTPUT`

##### `test_benchmark_output_not_empty()`
```python
def test_benchmark_output_not_empty():
    """Output must contain results"""
```
- **Depende de:** `test_benchmark_runs_without_crash()`
- **Valida:** Output tiene contenido (no vacío)

---

#### **Categoría 2: Performance Metrics (4 tests)**

##### `test_gpu_speedup_significant()`
```python
def test_gpu_speedup_significant():
    """GPU must show >2x speedup vs CPU"""
```
- **Valida:** Speedup >= 2.0x
- **Esperado:** 6-10x en RTX 5090
- **Crítico:** 🔴 HIGH (principal objetivo)

##### `test_cpu_throughput_reasonable()`
```python
def test_cpu_throughput_reasonable():
    """CPU baseline 5-200 docs/sec"""
```
- **Valida:** 5 <= throughput_cpu <= 200
- **Esperado:** 30-50 docs/sec
- **Crítico:** ⚠️ Medium (baseline reference)

##### `test_gpu_throughput_high()`
```python
def test_gpu_throughput_high():
    """GPU throughput >50 docs/sec"""
```
- **Valida:** throughput_gpu > 50
- **Esperado:** 200-400 docs/sec (RTX 5090)
- **Crítico:** 🔴 HIGH

##### `test_batch_processing_scales()`
```python
def test_batch_processing_scales():
    """Large batch should show >200 docs/sec"""
```
- **Valida:** large_batch_throughput > 200
- **Esperado:** 2000-3000 docs/sec
- **Crítico:** ⚠️ Medium (optimización)

---

#### **Categoría 3: Output Quality (3 tests)**

##### `test_embeddings_shape_correct()`
```python
def test_embeddings_shape_correct():
    """Embeddings must have correct shape"""
```
- **Valida:** Shape mencionado en output (100, X) donde X = 384 o 1024
- **Crítico:** 🔴 HIGH (correctitud)

##### `test_both_models_executed()`
```python
def test_both_models_executed():
    """Both CPU and GPU tests must run"""
```
- **Valida:** Output contiene "TEST 1" (CPU) y "TEST 2" (GPU)
- **Crítico:** ⚠️ Medium

##### `test_success_message_present()`
```python
def test_success_message_present():
    """Success indicator must be present"""
```
- **Valida:** Output contiene "[EXITO]" o "[OK]"
- **Crítico:** 🟡 LOW (indicador cualitativo)

---

#### **Categoría 4: Resource Usage (3 tests)**

##### `test_gpu_memory_usage_reasonable()`
```python
def test_gpu_memory_usage_reasonable():
    """GPU memory usage < 16GB"""
```
- **Valida:** VRAM usado < 16GB
- **Esperado:** 2-6GB
- **Crítico:** ⚠️ Medium

##### `test_no_errors_in_output()`
```python
def test_no_errors_in_output():
    """No error messages in output"""
```
- **Valida:** No contiene "[ERROR]" o "Traceback"
- **Crítico:** 🔴 HIGH

##### `test_completion_message()`
```python
def test_completion_message():
    """Completion message must be present"""
```
- **Valida:** Output contiene "COMPLETADO"
- **Crítico:** 🟡 LOW

---

### Variables Globales Compartidas

```python
# Estos datos se comparten entre todos los tests
BENCHMARK_OUTPUT = None        # String completo del output
BENCHMARK_DURATION = None      # Segundos de ejecución
BENCHMARK_EXIT_CODE = None     # 0 = success, >0 = error
```

**Uso:**
```python
# En tus tests puedes acceder:
def test_custom():
    assert BENCHMARK_OUTPUT is not None
    if "GPU" in BENCHMARK_OUTPUT:
        # Parse specific metrics
        pass
```

---

## 🔗 INTEGRATION TESTS: Pipeline Validation

### Archivo: `tests/integration/test_embedding_pipeline_20251002.py`

**Propósito:** Validar que los componentes del pipeline de embeddings funcionan correctamente juntos

**Cuándo ejecutar:**
- ✅ Después de cambios en código de embeddings
- ✅ Para validar edge cases y robustez
- ✅ Parte de ASSESS phase (opcional, profundo)

**Ejecución:**
```bash
# Opción 1: Todos los tests (LENTO - descarga modelos)
pytest tests/integration/test_embedding_pipeline_20251002.py -v

# Opción 2: Solo tests rápidos (sin descarga)
pytest tests/integration/test_embedding_pipeline_20251002.py -k "not load" -v

# Opción 3: Solo edge cases
pytest tests/integration/test_embedding_pipeline_20251002.py -k "edge or error" -v
```

---

### Tests Incluidos (16 total)

#### **Categoría 1: Model Loading (2 tests)**

##### `test_sentence_transformer_loads()`
```python
@pytest.fixture(scope="module")
def test_sentence_transformer_loads():
    """Load multilingual model successfully"""
```
- **Tipo:** Fixture (compartido por otros tests)
- **Acción:** Carga `paraphrase-multilingual-MiniLM-L12-v2`
- **Duración:** ~5-10 seg (primera vez desde caché)

##### `test_sentence_transformer_device_placement()`
```python
def test_sentence_transformer_device_placement(model):
    """Model can be moved to GPU"""
```
- **Valida:** `model.to('cuda:0')` funciona
- **Crítico:** 🔴 HIGH (GPU acceleration)

---

#### **Categoría 2: Encoding Functionality (4 tests)**

##### `test_model_encode_single_text(model)`
```python
def test_model_encode_single_text(model):
    """Encode single document"""
```
- **Valida:** Encode de 1 texto funciona
- **Output:** Shape (384,)

##### `test_model_encode_multiple_texts(model)`
```python
def test_model_encode_multiple_texts(model):
    """Encode batch of documents"""
```
- **Valida:** Encode de lista funciona
- **Output:** Shape (N, 384)

##### `test_model_encode_empty_list(model)`
```python
def test_model_encode_empty_list(model):
    """Handle empty input gracefully"""
```
- **Valida:** No crash con lista vacía
- **Edge case:** []

##### `test_model_encode_special_characters(model)`
```python
def test_model_encode_special_characters(model):
    """Handle Unicode and accents"""
```
- **Valida:** Textos con ñ, á, emojis
- **Edge case:** "Año español 🇪🇸"

---

#### **Categoría 3: Output Quality (4 tests)**

##### `test_embeddings_are_normalized(model)`
```python
def test_embeddings_are_normalized(model):
    """Embeddings should be L2 normalized"""
```
- **Valida:** `np.linalg.norm(embedding) ≈ 1.0`
- **Importante:** Para cosine similarity

##### `test_embeddings_dimension_consistent(model)`
```python
def test_embeddings_dimension_consistent(model):
    """All embeddings same dimension"""
```
- **Valida:** Todos los outputs tienen shape[1] == 384

##### `test_embeddings_reproducible(model)`
```python
def test_embeddings_reproducible(model):
    """Same input → same output"""
```
- **Valida:** Encode 2 veces da mismo resultado
- **Seed:** `torch.manual_seed(42)`

##### `test_embeddings_capture_similarity(model)`
```python
def test_embeddings_capture_similarity(model):
    """Similar texts have high cosine similarity"""
```
- **Valida:** Similarity > 0.7 para textos similares
- **Ejemplo:** "perro" vs "can" > 0.7

---

#### **Categoría 4: Batch Processing (3 tests)**

##### `test_batch_size_parameter(model)`
```python
def test_batch_size_parameter(model):
    """Batch size parameter works"""
```
- **Valida:** `encode(texts, batch_size=16)` funciona

##### `test_batch_vs_sequential_consistency(model)`
```python
def test_batch_vs_sequential_consistency(model):
    """Batch and sequential give same results"""
```
- **Valida:** Batch encoding == loop de single encodes

##### `test_large_batch_performance(model)`
```python
def test_large_batch_performance(model):
    """Large batch (1000 docs) completes"""
```
- **Valida:** 1000 documentos se procesan sin crash
- **Duración:** ~5-10 seg

---

#### **Categoría 5: GPU Acceleration (1 test)**

##### `test_gpu_acceleration_functional(model)`
```python
def test_gpu_acceleration_functional(model):
    """GPU encoding faster than CPU"""
```
- **Valida:** GPU > CPU en velocidad
- **Speedup esperado:** >2x

---

#### **Categoría 6: Error Handling (2 tests)**

##### `test_invalid_model_name_handling()`
```python
def test_invalid_model_name_handling():
    """Invalid model name raises error"""
```
- **Valida:** Exception con modelo inexistente
- **Edge case:** "nonexistent/fake-model-xyz"

##### `test_null_input_handling(model)`
```python
def test_null_input_handling(model):
    """Null/None inputs handled"""
```
- **Valida:** No crash con None en lista
- **Edge case:** [None, "text", None]

---

### Fixtures Disponibles

```python
@pytest.fixture(scope="module")
def model():
    """Shared model instance for all tests"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

**Ventaja:** Modelo se carga 1 sola vez, todos los tests lo reusan

---

## 🚀 Guía de Uso Rápida

### Workflow Completo (RPVEA-A)

```bash
# PASO 1: PREPARE - Generar tests (ya hecho por @test-architect)
# ✅ Tests ya existen

# PASO 2: VALIDATE - Ejecutar PRE-tests
python tests/pre/test_embedding_prereqs_20251002.py

# Si PRE-tests pasan:
# PASO 3: EXECUTE - Ejecutar benchmark
python benchmarks/scripts/embedding_benchmark.py | tee benchmarks/results/raw/embedding_output_20251002.txt

# PASO 4: ASSESS - Ejecutar POST-tests
python tests/post/test_embedding_success_20251002.py

# PASO 5: (Opcional) Integration tests
pytest tests/integration/test_embedding_pipeline_20251002.py -v
```

### Solo Validar Environment (Rápido)

```bash
# Solo tests críticos (< 30 seg)
pytest tests/pre/test_embedding_prereqs_20251002.py -k "cuda or gpu or packages" -v
```

### Debugging Tests Fallidos

```bash
# Ver output completo con prints
pytest tests/pre/test_embedding_prereqs_20251002.py -v -s

# Parar en primer fallo
pytest tests/pre/test_embedding_prereqs_20251002.py -x

# Ver traceback completo
pytest tests/pre/test_embedding_prereqs_20251002.py --tb=long
```

---

## 📊 Métricas de Testing

### Coverage Summary

| Test Suite | Tests | Categorías | Duración Estimada |
|-------------|-------|------------|-------------------|
| PRE-tests | 17 | 5 (Environment, GPU, Models, Dependencies, FS) | 30-60 seg |
| POST-tests | 12 | 4 (Execution, Performance, Quality, Resources) | 15-20 min* |
| Integration | 16 | 6 (Loading, Encoding, Quality, Batch, GPU, Errors) | 2-5 min* |

**\*Duración:** Incluye tiempo de ejecución del benchmark/cargas de modelo

### Test Criticality Distribution

| Nivel | PRE | POST | Integration | Total |
|-------|-----|------|-------------|-------|
| 🔴 HIGH | 7 | 4 | 3 | 14 (31%) |
| ⚠️ MEDIUM | 6 | 5 | 2 | 13 (29%) |
| 🟡 LOW | 4 | 3 | 11 | 18 (40%) |

---

## 🔧 Troubleshooting Común

### PRE-test: test_model_multilingual_exists() falla con 401

**Problema:** HF API requiere autenticación

**Soluciones:**
1. **Ignorar si modelo en caché:** Verificar con `ls ~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`
2. **Autenticar:** `huggingface-cli login`
3. **Modificar test:** Cambiar validación API por carga local

### POST-test tarda mucho (>30 min)

**Problema:** Primera ejecución descarga modelos

**Soluciones:**
1. **Pre-descargar modelos:**
   ```python
   from sentence_transformers import SentenceTransformer
   SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
   SentenceTransformer('dariolopez/bge-m3-es-legal-tmp-6')
   ```
2. **Skip POST-tests:** Ejecutar benchmark manualmente primero

### Integration tests fallan con "model not found"

**Problema:** Fixture no pudo cargar modelo

**Soluciones:**
1. Verificar internet conectado (primera descarga)
2. Limpiar caché corrupto: `rm -rf ~/.cache/huggingface/hub`
3. Re-instalar: `pip install --upgrade sentence-transformers`

---

## 📝 Changelog

**2025-10-02:**
- Suite de tests generada por @test-architect
- 45 tests totales (17 PRE, 12 POST, 16 Integration)
- Cobertura completa de embedding benchmark pipeline
- Documentación inicial de reference

**Futuras Expansiones:**
- [ ] Tests para LLM inference benchmarks
- [ ] Tests para dual-GPU configurations
- [ ] Tests para quantization comparisons
- [ ] Performance regression tests (comparar runs históricos)

---

## 🔗 Referencias

**Archivos de Tests:**
- PRE: `C:\Users\Gamer\Dev\LLM-Local-Lab\tests\pre\test_embedding_prereqs_20251002.py`
- POST: `C:\Users\Gamer\Dev\LLM-Local-Lab\tests\post\test_embedding_success_20251002.py`
- Integration: `C:\Users\Gamer\Dev\LLM-Local-Lab\tests\integration\test_embedding_pipeline_20251002.py`

**Documentación Relacionada:**
- Testing Strategy: `docs/testing/strategy_embedding_benchmark_20251002.md`
- RPVEA-A Methodology: `docs/workflows/rpvea-agent-integration.md`
- @test-architect Agent: `.claude/agents/test-architect.md`

**Mantenido por:** LLM-Local-Lab (RPVEA-A methodology)
