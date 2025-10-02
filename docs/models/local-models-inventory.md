# Inventario de Modelos Locales - LLM-Local-Lab

**Última actualización:** 2025-10-02
**Sistema:** AMD Ryzen 9 9950X + Dual RTX 5090 (64GB VRAM)
**Propósito:** Registro centralizado de todos los modelos disponibles localmente

---

## 🎯 Resumen Ejecutivo

**Total de Modelos Disponibles:**
- **LLMs (Ollama):** 5 modelos (89 GB total)
- **Embeddings (HF Cache):** 2 modelos (descargados, ~500MB)
- **Embeddings (Ollama):** 1 modelo configurado (nomic-embed-text)

**Storage Usado:** ~89.5 GB
**Storage Disponible:** 3.63 TB (suficiente para expansión)

---

## 🤖 SECCIÓN 1: LLMs (Ollama)

### Modelos Instalados

| Nombre | ID | Tamaño | Modificado | Cuantización | Parámetros | VRAM Estimado |
|--------|----|----|------------|--------------|------------|---------------|
| gemma2:27b | 53261bc9c192 | 15 GB | 2025-08-27 | Q4 (estimado) | 27B | ~18GB |
| hhao/qwen2.5-coder-tools:32b | 5d17b48771de | 19 GB | 2025-08-27 | Q4 (estimado) | 32B | ~22GB |
| qwen3-coder:30b | ad67f85ca250 | 18 GB | 2025-08-22 | Q4 (estimado) | 30B | ~20GB |
| deepseek-r1:8b | 6995872bfe4c | 5.2 GB | 2025-08-22 | Q4 (estimado) | 8B | ~8GB |
| deepseek-r1:latest | 6995872bfe4c | 5.2 GB | 2025-08-22 | Q4 (estimado) | 8B | ~8GB (alias) |

**Notas:**
- `deepseek-r1:8b` y `deepseek-r1:latest` apuntan al mismo modelo (mismo ID)
- Todos caben en una sola RTX 5090 (32GB VRAM)
- Cuantización estimada basada en ratio tamaño/parámetros

### Capacidades por Modelo

#### **gemma2:27b**
- **Especialización:** General-purpose, multimodal
- **Contexto:** 8K tokens (verificar con Ollama)
- **Velocidad estimada:** 15-25 tokens/sec (single GPU)
- **Mejor para:** Tareas generales, conversación, razonamiento

#### **hhao/qwen2.5-coder-tools:32b**
- **Especialización:** Code generation, tool use
- **Contexto:** 32K tokens (verificar)
- **Velocidad estimada:** 12-20 tokens/sec
- **Mejor para:** Generación de código, debugging, análisis técnico
- **Características especiales:** Tool calling integrado

#### **qwen3-coder:30b**
- **Especialización:** Code generation
- **Contexto:** 32K tokens (verificar)
- **Velocidad estimada:** 13-22 tokens/sec
- **Mejor para:** Generación de código puro, completion

#### **deepseek-r1:8b**
- **Especialización:** Razonamiento, chain-of-thought
- **Contexto:** 4K-8K tokens (verificar)
- **Velocidad estimada:** 40-60 tokens/sec
- **Mejor para:** Razonamiento complejo, problemas matemáticos, análisis lógico

### Acceso a Modelos Ollama

#### **Método 1: API Directa (Python)**
```python
import requests

OLLAMA_BASE_URL = "http://localhost:11434"

def query_ollama(model: str, prompt: str):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Ejemplo:
result = query_ollama("deepseek-r1:8b", "Explain RPVEA methodology")
```

#### **Método 2: LangChain Integration**
```python
from langchain_community.llms import Ollama

llm = Ollama(
    base_url="http://localhost:11434",
    model="gemma2:27b",
    temperature=0.7
)

response = llm.invoke("Your prompt here")
```

#### **Método 3: CLI (Testing rápido)**
```bash
# Chat interactivo
ollama run deepseek-r1:8b

# Single prompt
ollama run gemma2:27b "Explain quantum computing"

# Con parámetros
ollama run qwen3-coder:30b --temperature 0.2 "Write Python function to sort array"
```

### Verificación de Estado

```bash
# Listar modelos
ollama list

# Información detallada
ollama show gemma2:27b

# Verificar servicio activo
curl http://localhost:11434/api/tags
```

### Benchmark Esperado (RTX 5090)

| Modelo | Tokens/sec (FP16) | Tokens/sec (Q4) | Latencia First Token |
|--------|-------------------|-----------------|----------------------|
| deepseek-r1:8b | 80-120 | 40-60 | 150-300ms |
| gemma2:27b | 25-40 | 15-25 | 300-500ms |
| qwen3-coder:30b | 22-35 | 13-22 | 350-550ms |
| qwen2.5-coder-tools:32b | 20-32 | 12-20 | 400-600ms |

**Nota:** Benchmarks son estimaciones. Verificar con pruebas reales.

---

## 🔤 SECCIÓN 2: Modelos de Embeddings

### Modelos en Hugging Face Cache

**Ubicación:** `C:\Users\Gamer\.cache\huggingface\hub\`

#### **Modelo 1: paraphrase-multilingual-MiniLM-L12-v2**

**Metadata:**
- **Proveedor:** sentence-transformers (oficial)
- **Ruta completa:** `C:\Users\Gamer\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d`
- **Descargado:** 2025-08-26 18:51
- **Tamaño:** ~120 MB (modelo completo)
- **Snapshot ID:** 86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d

**Especificaciones Técnicas:**
- **Dimensión embeddings:** 384
- **Max sequence length:** 128 tokens
- **Idiomas soportados:** 50+ (incluyendo español)
- **Normalización:** L2 (cosine similarity ready)
- **Velocidad (CPU):** 30-50 docs/sec
- **Velocidad (GPU RTX 5090):** 200-400 docs/sec

**Uso Recomendado:**
- ✅ Baseline CPU (comparaciones de rendimiento)
- ✅ Textos cortos multilingües
- ✅ Búsqueda semántica general
- ⚠️ No especializado en legal/técnico

**Acceso (Python):**
```python
from sentence_transformers import SentenceTransformer

# Carga desde caché local (sin download)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# GPU
model = model.to('cuda:0')

# Generar embeddings
embeddings = model.encode([
    "Texto de ejemplo en español",
    "Another example in English"
])

# Output shape: (2, 384)
```

**Caso de Uso Original:**
- Usado en Langextract para embeddings generales
- Configurado como fallback/baseline

---

#### **Modelo 2: dariolopez/bge-m3-es-legal-tmp-6**

**Metadata:**
- **Proveedor:** dariolopez (fine-tuned)
- **Ruta completa:** `C:\Users\Gamer\.cache\huggingface\hub\models--dariolopez--bge-m3-es-legal-tmp-6\snapshots\42d0a03ceecf430ecfd7f3f49843b5dadb594bf9`
- **Descargado:** 2025-08-30 00:20
- **Tamaño:** ~350 MB (modelo completo)
- **Snapshot ID:** 42d0a03ceecf430ecfd7f3f49843b5dadb594bf9

**Especificaciones Técnicas:**
- **Base model:** BAAI/bge-m3 (fine-tuned para español legal)
- **Dimensión embeddings:** 1024
- **Max sequence length:** 512 tokens
- **Idioma principal:** Español (especializado legal)
- **Fine-tuning:** Corpus legal español (municipales, AAPP)
- **Velocidad (GPU RTX 5090):** 100-200 docs/sec (más pesado)

**Uso Recomendado:**
- ✅ Textos legales en español
- ✅ Documentos municipales/administrativos
- ✅ Búsqueda semántica especializada
- ⚠️ Requiere más VRAM (~2-4GB vs ~1GB del MiniLM)

**Acceso (Python):**
```python
from sentence_transformers import SentenceTransformer
import torch

# Carga desde caché local
model = SentenceTransformer('dariolopez/bge-m3-es-legal-tmp-6')

# Forzar GPU
device = torch.device('cuda:0')
model = model.to(device)

# Generar embeddings
legal_texts = [
    "La presente Ordenanza Municipal regula...",
    "El Ayuntamiento de Madrid establece..."
]

embeddings = model.encode(legal_texts)

# Output shape: (2, 1024)
```

**Caso de Uso Original:**
- Usado en Langextract para documentos legales municipales
- Especializado en corpus AAPP español

**Limitaciones:**
- ⚠️ Modelo temporal (`tmp-6` en nombre - puede ser inestable)
- ⚠️ Requiere más recursos que modelos base
- ⚠️ Acceso online requiere autenticación HF (caché local OK)

---

### Modelos Configurados en Ollama (No Descargados)

#### **nomic-embed-text:latest**

**Metadata:**
- **Configurado en:** Langextract config.py
- **Estado:** NO descargado en Ollama (solo referenciado)
- **Variable de entorno:** `OLLAMA_EMBEDDING_MODEL`

**Para Descargar:**
```bash
ollama pull nomic-embed-text
```

**Especificaciones (si se descarga):**
- **Dimensión:** 768
- **Max sequence:** 8192 tokens (contexto largo)
- **Idiomas:** Multilingüe (inglés optimizado)
- **Tamaño descarga:** ~274 MB

**Uso con Ollama API:**
```python
import requests

def get_ollama_embedding(text: str):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]
```

---

## 🔄 SECCIÓN 3: Integración con Proyectos Existentes

### Langextract - Configuración de Modelos

**Archivo:** `C:\Users\Gamer\Dev\Langextract\config.py`

**Embeddings configurados:**
```python
# Por defecto (sentence-transformers)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_PROVIDER = "sentence_transformers"
EMBEDDING_DIMENSION = 384

# Ollama (alternativa)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"

# OpenAI (cloud)
# text-embedding-3-small (1536 dim)
```

**LLMs configurados:**
```python
# Ollama local
OLLAMA_MODEL = "llama3.2:latest"  # ⚠️ No está en ollama list actual
OLLAMA_BASE_URL = "http://localhost:11434"

# LangExtract (OpenAI)
LANGEXTRACT_MODEL = "gpt-4o-mini"  # Requiere OPENAI_API_KEY
```

**Función de acceso:**
```python
from Langextract.config import get_embedding_config

config = get_embedding_config()
# Returns: {
#     "provider": "sentence_transformers",
#     "model": "paraphrase-multilingual-MiniLM-L12-v2",
#     "dimension": 384
# }
```

### RAG-Anything - Configuración

**Archivo:** `C:\Users\Gamer\Dev\RAG-Anything\raganything\config.py`

**Estado:** Archivo no encontrado en rutas esperadas (verificar estructura)

**TODO:** Investigar configuración de modelos en RAG-Anything

---

## 📋 SECCIÓN 4: Guía Rápida de Decisiones

### ¿Qué Modelo de Embeddings Usar?

```
┌─────────────────────────────────────────┐
│ ¿Qué tipo de documentos?               │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
    Legal Español       General/Multilingüe
        │                   │
        ▼                   ▼
   bge-m3-es-legal    paraphrase-multilingual
   (1024 dim)         (384 dim)
   ~200 docs/sec      ~300 docs/sec
   VRAM: 2-4GB        VRAM: 1GB
```

### ¿Qué LLM Usar?

```
┌─────────────────────────────────────────┐
│ ¿Qué tarea?                            │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
    Código              Razonamiento/Chat
        │                   │
        ▼                   ▼
    ┌───────┐           ┌──────────┐
    │ Speed │           │ Quality  │
    └───┬───┘           └────┬─────┘
        │                    │
        ▼                    ▼
   qwen3-coder:30b      gemma2:27b
   (13-22 tok/s)        (15-25 tok/s)

   Alternativa:         Alternativa:
   qwen2.5-coder-tools  deepseek-r1:8b
   (tool calling)       (40-60 tok/s, 8B)
```

---

## 🔧 SECCIÓN 5: Comandos de Mantenimiento

### Verificar Modelos HF Cache

```bash
# Listar todos los modelos en caché
ls -la "$HOME/.cache/huggingface/hub" | grep "models--"

# Ver tamaño de caché
du -sh "$HOME/.cache/huggingface/hub"

# Limpiar caché (CUIDADO - borra todo)
rm -rf "$HOME/.cache/huggingface/hub/*"
```

### Gestión de Modelos Ollama

```bash
# Listar modelos locales
ollama list

# Ver información detallada
ollama show gemma2:27b

# Eliminar modelo (liberar espacio)
ollama rm qwen3-coder:30b

# Actualizar modelo
ollama pull deepseek-r1:8b

# Verificar espacio usado
du -sh ~/.ollama/models
```

### Pre-descarga de Modelos HF

```python
from sentence_transformers import SentenceTransformer

# Forzar descarga si no existe
model = SentenceTransformer('BAAI/bge-m3')  # ~2.2GB download

# Verificar modelos en caché
from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()
for repo in cache_info.repos:
    print(f"{repo.repo_id}: {repo.size_on_disk / 1e9:.2f} GB")
```

---

## 📊 SECCIÓN 6: Benchmarks Planificados

### Embeddings (Próximos Tests)

| Modelo | Baseline CPU | Target GPU | Speedup Esperado |
|--------|--------------|------------|------------------|
| paraphrase-multilingual | 30-50 docs/s | 200-400 docs/s | 6-10x |
| bge-m3-es-legal | 15-30 docs/s | 100-200 docs/s | 5-8x |
| nomic-embed-text | TBD | TBD | TBD |

**Estado:**
- ✅ Script de benchmark creado: `benchmarks/scripts/embedding_benchmark.py`
- ⏳ Ejecución pendiente (RPVEA-A VALIDATE phase)

### LLMs (Futuros Tests)

**Prioridad Alta:**
1. deepseek-r1:8b - Baseline de velocidad (8B)
2. gemma2:27b - Modelo general intermedio
3. qwen3-coder:30b - Code generation benchmark

**Métricas a capturar:**
- Tokens/segundo (throughput)
- Time to first token (latencia)
- VRAM usage (peak y average)
- Context window real (test con prompts largos)
- Quality scores (perplexity, code correctness)

---

## 🚀 SECCIÓN 7: Próximos Pasos

### Modelos a Descargar (Recomendados)

**Embeddings:**
- [ ] `BAAI/bge-m3` (oficial, 2.2GB) - Alternativa a modelo tmp-6
- [ ] `intfloat/multilingual-e5-large` (2.2GB) - Benchmark comparison
- [ ] `nomic-embed-text` via Ollama (274MB) - Contexto largo

**LLMs:**
- [ ] `llama3.2:latest` (referenciado en Langextract pero no disponible)
- [ ] `llama3.1:8b` (baseline comparativo vs deepseek-r1)
- [ ] `codellama:13b` (alternativa code-focused)

### Documentación Pendiente

- [ ] Agregar benchmarks reales cuando se ejecuten
- [ ] Documentar configuraciones óptimas (batch size, context, etc.)
- [ ] Crear tabla de compatibilidad proyecto → modelo
- [ ] Agregar troubleshooting común

---

## 📝 Changelog

**2025-10-02:**
- Inventario inicial de modelos (5 LLMs Ollama, 2 embeddings HF)
- Documentación de rutas de acceso
- Integración con Langextract config
- Benchmarks estimados basados en hardware specs

---

## 🔗 Referencias

**Proyectos Relacionados:**
- Langextract: `C:\Users\Gamer\Dev\Langextract`
- RAG-Anything: `C:\Users\Gamer\Dev\RAG-Anything`

**Documentación Externa:**
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- sentence-transformers: https://www.sbert.net/
- Hugging Face Cache: https://huggingface.co/docs/huggingface_hub/guides/manage-cache

**Contacto:**
- Mantenido por: LLM-Local-Lab (este proyecto)
- Última revisión: Claude Code (RPVEA-A methodology)
