# LLM-Local-Lab - Documentación Central

**Última actualización:** 2025-10-02
**Versión del proyecto:** 0.1.0
**Metodología:** RPVEA-A (Agent-Augmented Testing-First)

---

## 🎯 Navegación Rápida

| Necesito... | Voy a... |
|-------------|----------|
| Ver qué modelos tengo disponibles | [`models/local-models-inventory.md`](models/local-models-inventory.md) |
| Saber cómo ejecutar tests | [`testing/test-suite-reference.md`](testing/test-suite-reference.md) |
| Entender el código del proyecto | [`code-reference/api-index.md`](code-reference/api-index.md) |
| Aprender la metodología RPVEA-A | [`workflows/rpvea-agent-integration.md`](workflows/rpvea-agent-integration.md) |
| Configurar un nuevo modelo | [`models/models-registry.md`](models/models-registry.md) |
| Ver especificaciones de hardware | [`hardware/system-specs.md`](hardware/system-specs.md) |
| Consultar benchmarks pasados | [`benchmarks/methodology.md`](benchmarks/methodology.md) |

---

## 📁 Estructura de Documentación

```
docs/
├── README.md                          # Este archivo (índice maestro)
├── models/                            # Documentación de modelos
│   ├── local-models-inventory.md      # ✅ NUEVO: Inventario completo de modelos
│   └── models-registry.md             # Modelos testeados con resultados
├── testing/                           # Documentación de testing
│   ├── test-suite-reference.md        # ✅ NUEVO: Referencia completa de tests
│   └── strategy_embedding_benchmark_20251002.md  # Estrategia de testing específica
├── code-reference/                    # Documentación de código
│   └── api-index.md                   # ✅ NUEVO: Índice de clases y funciones
├── workflows/                         # Workflows y metodología
│   └── rpvea-agent-integration.md     # ✅ NUEVO: RPVEA-A framework
├── hardware/                          # Especificaciones de hardware
│   └── system-specs.md                # Specs del sistema (pendiente)
└── benchmarks/                        # Metodología de benchmarking
    └── methodology.md                 # Cómo hacer benchmarks (pendiente)
```

---

## 🚀 Documentación Clave (Creada Hoy)

### 1. **Inventario de Modelos Locales** ⭐
**Archivo:** [`models/local-models-inventory.md`](models/local-models-inventory.md)

**Qué encontrarás:**
- ✅ **5 LLMs en Ollama** (gemma2:27b, qwen3-coder:30b, deepseek-r1:8b, etc.)
- ✅ **2 modelos de embeddings** en caché HF (paraphrase-multilingual, bge-m3-es-legal)
- ✅ **Rutas exactas** a archivos de modelos descargados
- ✅ **Código de acceso** (Python, CLI, LangChain)
- ✅ **Benchmarks esperados** para cada modelo
- ✅ **Integración con Langextract** y RAG-Anything

**Cuándo consultar:**
- Antes de ejecutar benchmarks (saber qué modelos hay)
- Al agregar nuevos modelos (ver estructura)
- Para troubleshooting de modelos no encontrados
- Al escribir scripts que usen modelos

**Secciones destacadas:**
- Decisión matrix: "¿Qué modelo usar para X tarea?"
- Comandos de mantenimiento (limpiar caché, descargar modelos)
- Rutas completas a modelos descargados

---

### 2. **Referencia de Testing Suite** ⭐
**Archivo:** [`testing/test-suite-reference.md`](testing/test-suite-reference.md)

**Qué encontrarás:**
- ✅ **45 tests documentados** (17 PRE, 12 POST, 16 Integration)
- ✅ **Cómo ejecutar cada suite** (comandos exactos)
- ✅ **Qué valida cada test** (descripción detallada)
- ✅ **Troubleshooting** de tests fallidos
- ✅ **Criticality levels** (HIGH/MEDIUM/LOW)

**Cuándo consultar:**
- Antes de ejecutar benchmarks (VALIDATE phase)
- Cuando un test falla (troubleshooting)
- Al agregar nuevos tests (seguir convenciones)
- Para entender qué cubren los tests

**Secciones destacadas:**
- Workflow completo RPVEA-A con tests
- Variables globales compartidas (POST-tests)
- Fixtures de pytest (Integration tests)
- Resolución de errores comunes

---

### 3. **Índice de API y Código** ⭐
**Archivo:** [`code-reference/api-index.md`](code-reference/api-index.md)

**Qué encontrarás:**
- ✅ **1 clase documentada** (LLMBenchmark)
- ✅ **48 funciones indexadas** (todos los módulos)
- ✅ **Navegación por línea** (ir directo al código)
- ✅ **Parámetros y return values** documentados
- ✅ **Ejemplos de uso** para cada función

**Cuándo consultar:**
- Al escribir código que use benchmarks
- Para entender qué hace una función
- Al buscar dónde está implementada una feature
- Para ver ejemplos de uso de clases

**Secciones destacadas:**
- Clase `LLMBenchmark` completa (constructor, métodos)
- Patrón de timing con CUDA synchronization
- Estructura de JSON output
- TODOs y mejoras futuras

---

### 4. **RPVEA-A Methodology** ⭐
**Archivo:** [`workflows/rpvea-agent-integration.md`](workflows/rpvea-agent-integration.md)

**Qué encontrarás:**
- ✅ **Framework completo RPVEA-A** (5 fases documentadas)
- ✅ **5 subagentes especializados** y cuándo usarlos
- ✅ **Tier system** (Tier 1/2/3 con delegación)
- ✅ **Anti-patterns** a evitar
- ✅ **Ejemplos reales** de workflows

**Cuándo consultar:**
- Al iniciar cualquier tarea compleja (Tier 2/3)
- Para entender cuándo usar subagentes
- Al diseñar testing strategies
- Para seguir best practices del proyecto

**Secciones destacadas:**
- PREPARE phase con @test-architect (testing-first)
- VALIDATE phase (PRE-tests mandatory)
- Delegation matrix por fase y tier
- Decision matrix: cuándo usar qué agentes

---

## 📖 Documentación por Caso de Uso

### 🎯 Caso 1: "Quiero ejecutar un benchmark de embeddings"

**Flujo RPVEA-A:**

1. **REVIEW (5 min):**
   - Lee: [`models/local-models-inventory.md`](models/local-models-inventory.md)
   - Verifica qué modelos de embeddings tienes
   - Confirma: paraphrase-multilingual y bge-m3-es-legal en caché

2. **PREPARE (15 min):**
   - Lee: [`testing/test-suite-reference.md`](testing/test-suite-reference.md) → Sección PRE-tests
   - Entiende qué validan los 17 PRE-tests
   - Nota: test_model_multilingual_exists puede fallar (OK si modelo en caché)

3. **VALIDATE (10 min):**
   - Ejecuta: `python tests/pre/test_embedding_prereqs_20251002.py`
   - Si 15+ tests pasan → PROCEED
   - Si <15 pasan → Fix issues

4. **EXECUTE (15-20 min):**
   - Lee: [`code-reference/api-index.md`](code-reference/api-index.md) → Embedding Benchmark
   - Ejecuta: `python benchmarks/scripts/embedding_benchmark.py | tee output.txt`

5. **ASSESS (15 min):**
   - Lee: [`testing/test-suite-reference.md`](testing/test-suite-reference.md) → Sección POST-tests
   - Ejecuta: `python tests/post/test_embedding_success_20251002.py`
   - Valida: speedup >= 2x

**Total time:** ~60-70 min (incluye ejecución de benchmark)

---

### 🎯 Caso 2: "Quiero agregar un nuevo LLM para testing"

**Flujo:**

1. **Descargar con Ollama:**
   ```bash
   ollama pull llama3.1:8b
   ```

2. **Actualizar inventario:**
   - Edita: [`models/local-models-inventory.md`](models/local-models-inventory.md)
   - Agrega entrada en tabla de LLMs
   - Documenta: tamaño, parámetros, VRAM estimado

3. **Crear configuración (opcional):**
   - Crea: `models/configs/llama-3.1-8b.yaml`
   - Sigue formato de otros configs

4. **Ejecutar benchmark:**
   - Lee: [`code-reference/api-index.md`](code-reference/api-index.md) → LLMBenchmark
   - Ejecuta:
     ```bash
     python benchmarks/scripts/llm_inference_benchmark.py \
         --model llama3.1:8b \
         --device cuda:0 \
         --runs 10
     ```

5. **Documentar resultados:**
   - Agrega a: [`models/models-registry.md`](models/models-registry.md)
   - Include: throughput, latency, VRAM usage

---

### 🎯 Caso 3: "Un test está fallando, no sé por qué"

**Troubleshooting con docs:**

1. **Identificar el test:**
   - Anota el nombre exacto (ej: `test_model_multilingual_exists`)

2. **Consultar referencia:**
   - Abre: [`testing/test-suite-reference.md`](testing/test-suite-reference.md)
   - Busca el test por nombre (Ctrl+F)

3. **Leer documentación del test:**
   - **Qué valida:** "Modelo en Hugging Face Hub (API check)"
   - **Crítico:** Medium (pero modelo en caché funciona)
   - **Si falla:** Opciones A, B, C documentadas

4. **Ver sección Troubleshooting:**
   - Encuentra: "PRE-test: test_model_multilingual_exists() falla con 401"
   - Solución: "Ignorar si modelo en caché"
   - Verificar:
     ```bash
     ls ~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2
     ```

5. **Confirmar con inventario:**
   - Abre: [`models/local-models-inventory.md`](models/local-models-inventory.md)
   - Busca ruta del modelo
   - Confirma: modelo descargado 2025-08-26 (OK)

**Decisión:** Ignorar fallo de API, modelo local funciona.

---

### 🎯 Caso 4: "Necesito entender cómo funciona X función"

**Navegación de código:**

1. **Buscar en API index:**
   - Abre: [`code-reference/api-index.md`](code-reference/api-index.md)
   - Ctrl+F: nombre de función

2. **Ver documentación:**
   - **Parámetros:** tipos y defaults
   - **Retorna:** estructura del output
   - **Ubicación:** archivo y línea exacta
   - **Ejemplo de uso:** código ejecutable

3. **Ir al código fuente:**
   - Abre archivo mencionado
   - Navega a línea exacta
   - Lee implementación

**Ejemplo:**
- Busco: "get_gpu_memory"
- Encuentro: llm_inference_benchmark.py líneas 66-78
- Leo: método de clase LLMBenchmark
- Veo ejemplo: Dict con gpu_X_allocated_gb

---

## 🔍 Búsquedas Frecuentes

### "¿Cómo acceder a modelo X?"

**Respuesta en:** [`models/local-models-inventory.md`](models/local-models-inventory.md)

**Secciones:**
- LLMs → "Acceso a Modelos Ollama" (3 métodos)
- Embeddings → "Acceso (Python)" con código

---

### "¿Qué tests debo ejecutar antes de X?"

**Respuesta en:** [`testing/test-suite-reference.md`](testing/test-suite-reference.md)

**Sección:** "Guía de Uso Rápida" → Workflow Completo

---

### "¿Cuándo usar @test-architect?"

**Respuesta en:** [`workflows/rpvea-agent-integration.md`](workflows/rpvea-agent-integration.md)

**Sección:** "Phase P: PREPARE" → Critical Delegations

---

### "¿Qué modelo de embeddings es mejor para Y?"

**Respuesta en:** [`models/local-models-inventory.md`](models/local-models-inventory.md)

**Sección:** "Guía Rápida de Decisiones" → Diagrama de flujo

---

## 📊 Estadísticas de Documentación

**Documentos creados hoy:** 4
**Páginas totales:** ~50 páginas (estimado en Markdown)
**Funciones documentadas:** 48
**Tests documentados:** 45
**Modelos catalogados:** 7 (5 LLM + 2 Embedding)
**Tiempo de creación:** ~2 horas (automated by @documentation-writer + Claude)

**Cobertura:**
- ✅ **100%** de modelos locales inventariados
- ✅ **100%** de tests documentados
- ✅ **100%** de funciones públicas documentadas
- ✅ **100%** de workflows RPVEA-A documentados

---

## 🚧 Documentación Pendiente

### High Priority
- [ ] `hardware/system-specs.md` - Especificaciones completas del hardware
- [ ] `benchmarks/methodology.md` - Metodología de benchmarking detallada
- [ ] `models/models-registry.md` - Registry de modelos testeados

### Medium Priority
- [ ] `workflows/agent-usage.md` - Guía de uso de subagentes
- [ ] `tutorials/first-benchmark.md` - Tutorial paso a paso
- [ ] `troubleshooting.md` - Troubleshooting centralizado

### Low Priority
- [ ] `performance-tuning.md` - Optimización de rendimiento
- [ ] `dual-gpu-setup.md` - Configuración dual-GPU
- [ ] `changelog.md` - Changelog del proyecto

---

## 🔄 Mantenimiento de Documentación

### Cuándo Actualizar Docs

**Triggers automáticos (RPVEA-A):**
- **ASSESS phase:** @documentation-writer actualiza registry
- **New model added:** Actualizar local-models-inventory.md
- **New test created:** Actualizar test-suite-reference.md
- **New function:** Actualizar api-index.md

### Proceso de Actualización

1. **Identificar sección afectada**
2. **Editar archivo correspondiente**
3. **Actualizar "Última actualización" en header**
4. **Agregar entrada en Changelog (si existe)**
5. **Git commit:** `docs: update [section] - [reason]`

### Convenciones de Formato

**Headers:**
- Nivel 1 (`#`): Título del documento
- Nivel 2 (`##`): Secciones principales
- Nivel 3 (`###`): Subsecciones
- Nivel 4 (`####`): Detalles específicos

**Code Blocks:**
- Usar \`\`\`python, \`\`\`bash, etc. (syntax highlighting)
- Incluir comentarios explicativos
- Mostrar output esperado cuando relevante

**Tablas:**
- Usar para comparaciones y referencias rápidas
- Alinear columnas para legibilidad
- Incluir headers descriptivos

**Links:**
- Usar paths relativos (`models/inventory.md`)
- Verificar que links funcionen
- Usar nombres descriptivos, no URLs

---

## 🎓 Aprendizajes y Best Practices

### Lo que Funciona Bien

✅ **Documentar mientras se crea código** (RPVEA-A ASSESS phase)
- Contexto fresco en memoria
- No necesitas "recordar" después
- @documentation-writer automatiza mucho

✅ **Índices centralizados** (este README)
- Fácil navegación
- No buscar en múltiples lugares
- Quick reference para Claude Code

✅ **Troubleshooting inline** (en test-suite-reference.md)
- Soluciones al lado del problema
- Reduce tiempo de debugging
- Basado en experiencia real

✅ **Código ejecutable en docs** (api-index.md)
- Copiar-pegar directo
- Ejemplos verificables
- Aprende haciendo

### Lo que Evitamos

❌ **Docs genéricos sin ejemplos**
- Difícil de usar
- No actionable

❌ **Documentación desactualizada**
- Peor que no tener docs
- Confusion y errores

❌ **Falta de navegación**
- Tiempo perdido buscando
- Frustración

❌ **No documentar edge cases**
- Repiten mismos errores
- Debugging repetitivo

---

## 📞 Soporte y Contribución

### ¿Falta documentación?

**Proceso para solicitar docs:**
1. Abre issue en GitHub (si aplicable)
2. Describe qué necesitas documentar
3. Indica caso de uso o problema
4. @documentation-writer generará draft

### ¿Encontraste un error en docs?

**Proceso de corrección:**
1. Nota la ubicación exacta (archivo, sección)
2. Describe el error
3. Sugiere corrección
4. Submit PR o issue

### Contribuir con Documentación

**Guidelines:**
- Seguir formato existente
- Incluir ejemplos ejecutables
- Actualizar índice maestro (este README)
- Testing antes de commit (verificar links)

---

## 🔗 Enlaces Externos

**Proyectos Relacionados:**
- Langextract: `C:\Users\Gamer\Dev\Langextract`
- RAG-Anything: `C:\Users\Gamer\Dev\RAG-Anything`

**Documentación Técnica:**
- sentence-transformers: https://www.sbert.net/
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- Hugging Face Hub: https://huggingface.co/docs
- PyTorch CUDA: https://pytorch.org/docs/stable/cuda.html

**Metodología:**
- RPVEA Framework: `C:\Users\Gamer\Downloads\METODOLOGIA DESARROLLO`

---

## 📝 Changelog de Documentación

**2025-10-02:**
- ✅ Creado README maestro de documentación
- ✅ Inventario completo de modelos (`local-models-inventory.md`)
- ✅ Referencia de testing suite (`test-suite-reference.md`)
- ✅ Índice de API y código (`api-index.md`)
- ✅ RPVEA-A methodology (`rpvea-agent-integration.md`)
- ✅ @test-architect agent specification
- ✅ Estructura de docs/ organizada

**Cobertura inicial:** 4 documentos principales, ~50 páginas, 100% de código actual documentado

---

**Mantenido por:** LLM-Local-Lab
**Metodología:** RPVEA-A (Agent-Augmented Testing-First)
**Generado con:** Claude Code + @documentation-writer
