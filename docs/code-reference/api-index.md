# API & Code Reference Index - LLM-Local-Lab

**Última actualización:** 2025-10-02
**Propósito:** Índice centralizado de clases, funciones y métodos del proyecto
**Audiencia:** Desarrolladores, Claude Code (para rápida navegación)

---

## 🎯 Resumen de Módulos

| Módulo | Archivo | Clases | Funciones | LOC |
|--------|---------|--------|-----------|-----|
| Embedding Benchmark | `benchmarks/scripts/embedding_benchmark.py` | 0 | 1 | 120 |
| LLM Inference Benchmark | `benchmarks/scripts/llm_inference_benchmark.py` | 1 | 2 | 281 |
| PRE-tests | `tests/pre/test_embedding_prereqs_20251002.py` | 0 | 17 | ~400 |
| POST-tests | `tests/post/test_embedding_success_20251002.py` | 0 | 12 | ~500 |
| Integration tests | `tests/integration/test_embedding_pipeline_20251002.py` | 0 | 16 | ~600 |

**Total:** 1 clase, 48 funciones, ~1901 LOC

---

## 📦 MÓDULO 1: Embedding Benchmark

**Archivo:** `benchmarks/scripts/embedding_benchmark.py`
**Propósito:** Benchmark de rendimiento CPU vs GPU para modelos de embeddings
**Tipo:** Script ejecutable
**Dependencias:** torch, sentence_transformers, numpy

### Funciones

#### `setup_utf8_encoding()`
```python
def setup_utf8_encoding():
    """Setup UTF-8 encoding for Windows compatibility"""
```
- **Parámetros:** None
- **Retorna:** None
- **Side effects:** Modifica `os.environ` (PYTHONIOENCODING, PYTHONUTF8)
- **Uso:** Llamar al inicio del script en Windows

**Ubicación:** Línea 10-13
**Invocaciones:** Línea 14 (auto-invocado)

---

### Script Main Flow

#### Sección 1: GPU Verification (Líneas 20-30)
```python
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```
- **Propósito:** Validar environment antes de tests
- **Output:** Info de sistema a stdout

#### Sección 2: Test Documents (Líneas 32-39)
```python
test_documents = [
    "La presente Ordenanza Municipal regula...",
    # ...
] * 20  # 100 documentos
```
- **Dataset:** 5 textos municipales españoles × 20 = 100 docs
- **Idioma:** Español (legal/administrativo)
- **Longitud promedio:** ~50-70 caracteres

#### Sección 3: CPU Baseline Test (Líneas 43-55)
```python
model_multilingual = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings_cpu = model_multilingual.encode(test_documents)
```
- **Modelo:** paraphrase-multilingual-MiniLM-L12-v2
- **Device:** CPU (por defecto)
- **Métricas:** tiempo total, docs/seg, shape

#### Sección 4: GPU Accelerated Test (Líneas 57-89)
```python
model_bge = SentenceTransformer('dariolopez/bge-m3-es-legal-tmp-6')
device = torch.device('cuda:0')
model_bge = model_bge.to(device)
embeddings_gpu = model_bge.encode(test_documents)
```
- **Modelo:** dariolopez/bge-m3-es-legal-tmp-6
- **Device:** cuda:0 (hardcoded)
- **Speedup calculation:** cpu_time / gpu_time

#### Sección 5: Memory Usage Test (Líneas 91-101)
```python
memory_before = torch.cuda.memory_allocated(0) / 1024**3
memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
```
- **GPU:** Hardcoded GPU 0
- **Units:** Gigabytes (GB)

#### Sección 6: Large Batch Test (Líneas 103-115)
```python
large_batch = test_documents * 10  # 1000 documentos
large_embeddings = model_bge.encode(large_batch, batch_size=32)
```
- **Batch size:** 1000 documentos
- **Batch parameter:** 32 (configurable)
- **Propósito:** Test de escalabilidad

**Output Final:** Líneas 117-120

---

## 📦 MÓDULO 2: LLM Inference Benchmark

**Archivo:** `benchmarks/scripts/llm_inference_benchmark.py`
**Propósito:** Benchmark de inferencia para LLMs (throughput, latency, VRAM)
**Tipo:** Script ejecutable + Clase reutilizable
**Dependencias:** torch, transformers, numpy, argparse

### Funciones Globales

#### `setup_utf8_encoding()`
```python
def setup_utf8_encoding():
    """Setup UTF-8 encoding for Windows compatibility"""
```
- **Idéntica a embedding_benchmark.py**
- **Ubicación:** Líneas 18-22

#### `main()`
```python
def main():
    """CLI entry point with argument parsing"""
```
- **Parámetros:** None (usa sys.argv)
- **Retorna:** None
- **Ubicación:** Líneas 172-277
- **CLI Args:**
  - `--model` (required): Hugging Face model name
  - `--device` (default: cuda:0): Device placement
  - `--quantization` (optional): 4bit, 8bit, or None
  - `--max-tokens` (default: 100): Max new tokens
  - `--warmup` (default: 3): Warmup runs
  - `--runs` (default: 10): Test runs
  - `--output` (optional): Output JSON path

**Ejemplo de uso:**
```bash
python llm_inference_benchmark.py \
    --model deepseek-r1:8b \
    --device cuda:0 \
    --quantization 4bit \
    --max-tokens 100 \
    --runs 10
```

---

### Clase: `LLMBenchmark`

**Ubicación:** Líneas 26-169
**Propósito:** Encapsular lógica de benchmarking de LLMs

#### Constructor: `__init__()`
```python
def __init__(self, model_name: str, device: str = "cuda:0", quantization: str = None):
```
- **Parámetros:**
  - `model_name`: Hugging Face model ID
  - `device`: torch device string (cuda:0, cuda:1, cpu)
  - `quantization`: "4bit", "8bit", or None
- **Atributos inicializados:**
  - `self.model`: None (loaded later)
  - `self.tokenizer`: None (loaded later)
- **Ubicación:** Líneas 29-35

---

#### Método: `load_model()`
```python
def load_model(self):
    """Load model and tokenizer"""
```
- **Parámetros:** None (usa self.model_name, etc.)
- **Retorna:** None (side effect: carga self.model y self.tokenizer)
- **Dependencias:** transformers.AutoModelForCausalLM, AutoTokenizer
- **Cuantización soportada:**
  - `None`: FP16 (torch.float16)
  - `"8bit"`: load_in_8bit=True
  - `"4bit"`: load_in_4bit=True
- **Device mapping:** Automático con `device_map`
- **Ubicación:** Líneas 37-64

**Uso típico:**
```python
benchmark = LLMBenchmark("deepseek-r1:8b", "cuda:0", "4bit")
benchmark.load_model()  # Descarga y carga modelo
```

---

#### Método: `get_gpu_memory()`
```python
def get_gpu_memory(self) -> Dict[str, float]:
    """Get current GPU memory usage"""
```
- **Parámetros:** None
- **Retorna:** Dict con métricas de VRAM por GPU
  - Keys: `gpu_{i}_allocated_gb`, `gpu_{i}_reserved_gb`
  - Values: float (GB)
- **Ejemplo output:**
  ```python
  {
      "gpu_0_allocated_gb": 4.25,
      "gpu_0_reserved_gb": 5.12,
      "gpu_1_allocated_gb": 0.0,
      "gpu_1_reserved_gb": 0.0
  }
  ```
- **Ubicación:** Líneas 66-78

---

#### Método: `run_inference()`
```python
def run_inference(self, prompt: str, max_new_tokens: int = 100) -> Dict[str, Any]:
    """Run single inference and measure metrics"""
```
- **Parámetros:**
  - `prompt`: Input text
  - `max_new_tokens`: Maximum tokens to generate
- **Retorna:** Dict con métricas de inferencia
  ```python
  {
      "input_tokens": int,        # Tokens en prompt
      "output_tokens": int,       # Tokens generados
      "total_tokens": int,        # input + output
      "total_time_s": float,      # Tiempo total (seg)
      "tokens_per_second": float, # Throughput
      "generated_text": str       # Primeros 100 chars
  }
  ```
- **Ubicación:** Líneas 80-119
- **Características:**
  - Usa `torch.cuda.synchronize()` para medición precisa
  - `do_sample=False` para reproducibilidad
  - Trunca texto generado a 100 caracteres

**Uso típico:**
```python
result = benchmark.run_inference("Explain quantum computing", max_new_tokens=50)
print(f"Speed: {result['tokens_per_second']} tok/s")
```

---

#### Método: `benchmark()`
```python
def benchmark(self, prompts: List[str], max_new_tokens: int = 100,
              warmup_runs: int = 3, test_runs: int = 10) -> Dict[str, Any]:
    """Run full benchmark with warmup and multiple test runs"""
```
- **Parámetros:**
  - `prompts`: Lista de prompts de test
  - `max_new_tokens`: Tokens a generar por run
  - `warmup_runs`: Runs de calentamiento (descartados)
  - `test_runs`: Runs de medición
- **Retorna:** Dict con resultados agregados
  ```python
  {
      "stats": {
          "throughput_mean": float,
          "throughput_std": float,
          "throughput_min": float,
          "throughput_max": float,
          "latency_mean_s": float,
          "latency_std_s": float
      },
      "individual_runs": List[Dict],  # Resultados por run
      "gpu_memory": Dict              # Métricas de VRAM
  }
  ```
- **Proceso:**
  1. Warmup runs (para estabilizar GPU)
  2. Clear CUDA cache
  3. Test runs con medición
  4. Cálculo de estadísticas (mean, std, min, max)
- **Ubicación:** Líneas 121-169

**Uso típico:**
```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
results = benchmark.benchmark(prompts, max_new_tokens=100, test_runs=10)
print(f"Mean throughput: {results['stats']['throughput_mean']} tok/s")
```

---

## 🧪 MÓDULO 3: Test Suites

**Ver documentación completa en:** `docs/testing/test-suite-reference.md`

### PRE-tests (`tests/pre/test_embedding_prereqs_20251002.py`)

**17 funciones de test:**

1. `test_python_version()` - Python >= 3.10
2. `test_required_packages_installed()` - torch, sentence_transformers, etc.
3. `test_cuda_available()` - CUDA disponible
4. `test_gpu_count()` - Al menos 1 GPU
5. `test_gpu_memory_sufficient()` - >= 8GB VRAM
6. `test_gpu_idle()` - Utilización < 50%
7. `test_gpu_compute_capability()` - Compute >= 7.0
8. `test_sentence_transformers_installed()` - Package importable
9. `test_model_multilingual_exists()` - Modelo en HF Hub
10. `test_model_bge_legal_exists()` - Modelo BGE disponible
11. `test_torch_version()` - PyTorch >= 2.0
12. `test_numpy_available()` - NumPy importable
13. `test_transformers_available()` - transformers importable
14. `test_accelerate_available()` - accelerate importable
15. `test_benchmark_script_exists()` - Script existe
16. `test_results_directory_writable()` - Directorio escribible
17. `test_disk_space_sufficient()` - >= 10GB libres

**Ejecución:** `python tests/pre/test_embedding_prereqs_20251002.py`

---

### POST-tests (`tests/post/test_embedding_success_20251002.py`)

**12 funciones de test:**

1. `test_benchmark_runs_without_crash()` - **EJECUTA BENCHMARK**
2. `test_benchmark_output_not_empty()` - Output no vacío
3. `test_gpu_speedup_significant()` - Speedup >= 2x
4. `test_cpu_throughput_reasonable()` - 5-200 docs/sec
5. `test_gpu_throughput_high()` - > 50 docs/sec
6. `test_batch_processing_scales()` - Batch > 200 docs/sec
7. `test_embeddings_shape_correct()` - Shape (100, X)
8. `test_both_models_executed()` - CPU y GPU tests corrieron
9. `test_success_message_present()` - "[EXITO]" en output
10. `test_gpu_memory_usage_reasonable()` - < 16GB VRAM
11. `test_no_errors_in_output()` - Sin "[ERROR]" o tracebacks
12. `test_completion_message()` - "COMPLETADO" presente

**Variables globales compartidas:**
- `BENCHMARK_OUTPUT`: String con output completo
- `BENCHMARK_DURATION`: Segundos de ejecución
- `BENCHMARK_EXIT_CODE`: 0 = success

**Ejecución:** `python tests/post/test_embedding_success_20251002.py`

---

### Integration tests (`tests/integration/test_embedding_pipeline_20251002.py`)

**16 funciones de test + 1 fixture:**

**Fixture:**
- `model()` - Carga SentenceTransformer (scope="module")

**Tests de Loading:**
1. `test_sentence_transformer_loads()` - Carga exitosa
2. `test_sentence_transformer_device_placement()` - GPU placement

**Tests de Encoding:**
3. `test_model_encode_single_text()` - Single doc
4. `test_model_encode_multiple_texts()` - Batch
5. `test_model_encode_empty_list()` - Edge case: []
6. `test_model_encode_special_characters()` - Unicode, ñ, emojis

**Tests de Quality:**
7. `test_embeddings_are_normalized()` - L2 norm ≈ 1.0
8. `test_embeddings_dimension_consistent()` - Dim = 384
9. `test_embeddings_reproducible()` - Deterministic
10. `test_embeddings_capture_similarity()` - Semantic similarity

**Tests de Batch:**
11. `test_batch_size_parameter()` - batch_size funciona
12. `test_batch_vs_sequential_consistency()` - Batch == loop
13. `test_large_batch_performance()` - 1000 docs

**Tests de GPU:**
14. `test_gpu_acceleration_functional()` - GPU > CPU

**Tests de Errors:**
15. `test_invalid_model_name_handling()` - Exception con modelo fake
16. `test_null_input_handling()` - No crash con None

**Ejecución:** `pytest tests/integration/test_embedding_pipeline_20251002.py -v`

---

## 🔧 Utilidades Comunes

### Encoding UTF-8 Setup

**Función:** `setup_utf8_encoding()`
**Archivos:** embedding_benchmark.py, llm_inference_benchmark.py
**Ubicación:** Líneas 10-13 (embedding), 18-22 (llm)

```python
def setup_utf8_encoding():
    """Setup UTF-8 encoding for Windows compatibility"""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
```

**Uso:** Llamar al inicio de cualquier script con output Unicode en Windows

**Por qué es necesario:**
- Windows usa encoding diferente por defecto (cp1252)
- Modelos multilingües generan caracteres Unicode
- Evita `UnicodeEncodeError` en prints

---

### GPU Memory Tracking

**Patrón común en benchmarks:**

```python
# Antes del test
torch.cuda.empty_cache()
memory_before = torch.cuda.memory_allocated(0) / 1024**3

# ... operación ...

# Después del test
memory_after = torch.cuda.memory_allocated(0) / 1024**3
memory_used = memory_after - memory_before
```

**Ubicaciones:**
- embedding_benchmark.py: Líneas 94-100
- llm_inference_benchmark.py: Método `get_gpu_memory()` (66-78)

---

### Timing Precisión

**Patrón con CUDA synchronization:**

```python
torch.cuda.synchronize() if torch.cuda.is_available() else None
start_time = time.perf_counter()

# ... operación GPU ...

torch.cuda.synchronize() if torch.cuda.is_available() else None
end_time = time.perf_counter()
total_time = end_time - start_time
```

**Por qué:**
- Operaciones CUDA son asíncronas
- Sin `synchronize()`, timing sería incorrecto
- `time.perf_counter()` es más preciso que `time.time()`

**Ubicaciones:**
- llm_inference_benchmark.py: Líneas 88-101

---

## 📊 Data Structures

### Benchmark Output Format (LLM Inference)

**Archivo JSON generado por llm_inference_benchmark.py:**

```json
{
  "timestamp": "2025-10-02T14:30:00",
  "model": "deepseek-r1:8b",
  "device": "cuda:0",
  "quantization": "4bit",
  "max_new_tokens": 100,
  "warmup_runs": 3,
  "test_runs": 10,
  "results": {
    "stats": {
      "throughput_mean": 45.23,
      "throughput_std": 2.15,
      "throughput_min": 42.10,
      "throughput_max": 48.50,
      "latency_mean_s": 2.21,
      "latency_std_s": 0.11
    },
    "individual_runs": [
      {
        "input_tokens": 15,
        "output_tokens": 100,
        "total_tokens": 115,
        "total_time_s": 2.23,
        "tokens_per_second": 44.84,
        "generated_text": "Machine learning is..."
      }
    ],
    "gpu_memory": {
      "gpu_0_allocated_gb": 4.25,
      "gpu_0_reserved_gb": 5.12
    }
  },
  "system_info": {
    "pytorch_version": "2.9.0.dev20250829+cu128",
    "cuda_available": true,
    "cuda_version": "13.0",
    "gpu_count": 2
  }
}
```

**Ubicación:** Líneas 245-261 (llm_inference_benchmark.py)

---

## 🔍 Navegación Rápida

### Encontrar Implementación de Feature

| Feature | Archivo | Línea(s) | Tipo |
|---------|---------|----------|------|
| Embedding CPU baseline | embedding_benchmark.py | 43-55 | Code |
| Embedding GPU test | embedding_benchmark.py | 57-89 | Code |
| LLM quantization | llm_inference_benchmark.py | 54-57 | Code |
| GPU memory tracking | llm_inference_benchmark.py | 66-78 | Method |
| Benchmark warmup | llm_inference_benchmark.py | 131-136 | Code |
| Results JSON save | llm_inference_benchmark.py | 245-273 | Code |
| PRE-tests GPU validation | test_embedding_prereqs_20251002.py | ~100-180 | Tests |
| POST-tests speedup check | test_embedding_success_20251002.py | ~120-150 | Tests |

---

### Parámetros Configurables

**embedding_benchmark.py:**
- ❌ **No configurable** (hardcoded values)
- Modelos: Líneas 45, 60
- Device: Línea 64 (cuda:0)
- Batch size: Línea 108 (32)
- Documentos de test: Líneas 33-39

**llm_inference_benchmark.py:**
- ✅ **Configurable vía CLI**
- Ver `main()` argparse (Líneas 173-188)
- Defaults: device=cuda:0, max_tokens=100, warmup=3, runs=10

---

## 🚧 TODOs y Mejoras Futuras

### Embedding Benchmark
- [ ] Agregar argumentos CLI (actualmente hardcoded)
- [ ] Salida JSON estructurada (actualmente solo stdout)
- [ ] Soporte dual-GPU (actualmente solo cuda:0)
- [ ] Configuración de batch_size como parámetro
- [ ] Métricas adicionales: perplexity, semantic similarity tests

### LLM Inference Benchmark
- [ ] Soporte para Ollama (actualmente solo HF)
- [ ] Benchmark de dual-GPU parallelization
- [ ] Context window tests (2K, 4K, 8K, 32K)
- [ ] Quality metrics (beyond throughput/latency)
- [ ] Comparative mode (benchmark múltiples modelos en un run)

### Testing
- [ ] Tests para llm_inference_benchmark.py (solo embeddings por ahora)
- [ ] CI/CD integration (GitHub Actions)
- [ ] Regression tests (comparar con baselines históricos)

---

## 📝 Changelog

**2025-10-02:**
- Documentación inicial de API
- 2 módulos de benchmark documentados
- 3 test suites catalogadas
- 48 funciones indexadas

**Próximas actualizaciones:**
- Agregar utils/ cuando se creen
- Documentar configs/ cuando existan archivos YAML
- Índice de hooks cuando se implementen

---

## 🔗 Referencias

**Archivos Fuente:**
- `benchmarks/scripts/embedding_benchmark.py`
- `benchmarks/scripts/llm_inference_benchmark.py`
- `tests/pre/test_embedding_prereqs_20251002.py`
- `tests/post/test_embedding_success_20251002.py`
- `tests/integration/test_embedding_pipeline_20251002.py`

**Documentación Relacionada:**
- Test Suite Reference: `docs/testing/test-suite-reference.md`
- Local Models Inventory: `docs/models/local-models-inventory.md`
- RPVEA-A Methodology: `docs/workflows/rpvea-agent-integration.md`

**Mantenido por:** LLM-Local-Lab (RPVEA-A methodology)
