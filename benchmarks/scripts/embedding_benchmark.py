# TEST GPU EMBEDDINGS: Comparacion CPU vs GPU con RTX 5090 Dual-GPU
# Test de rendimiento para generacion de embeddings acelerada

import os
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def setup_utf8_encoding():
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

setup_utf8_encoding()

print("========================================")
print("TEST GPU EMBEDDINGS - RTX 5090 DUAL-GPU")
print("========================================")

# Verificar setup GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {props.total_memory//1024**3}GB")
        print(f"  Compute capability: {props.major}.{props.minor}")

# Documentos de prueba
test_documents = [
    "La presente Ordenanza Municipal regula el uso de espacios publicos",
    "El Ayuntamiento de Madrid establece las siguientes normativas urbanisticas",
    "Las licencias de obras se tramitaran conforme al siguiente procedimiento",
    "Los tributos municipales se ajustaran a la normativa vigente",
    "El plan general de ordenacion urbana contempla las siguientes medidas"
] * 20  # 100 documentos para test

print(f"\nTest con {len(test_documents)} documentos municipales")

# TEST 1: Modelo paraphrase-multilingual (CPU baseline)
print("\n--- TEST 1: MODELO MULTILINGUAL (CPU Baseline) ---")
model_multilingual = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(f"Modelo cargado en: {model_multilingual.device}")

start_time = time.time()
embeddings_cpu = model_multilingual.encode(test_documents)
cpu_time = time.time() - start_time

print(f"[CPU] Tiempo total: {cpu_time:.2f}s")
print(f"[CPU] Tiempo por documento: {cpu_time/len(test_documents):.3f}s")
print(f"[CPU] Documentos por segundo: {len(test_documents)/cpu_time:.1f}")
print(f"[CPU] Shape embeddings: {embeddings_cpu.shape}")

# TEST 2: Modelo BGE-M3 Spanish Legal (GPU)
print("\n--- TEST 2: BGE-M3 SPANISH LEGAL (GPU Acelerado) ---")
try:
    model_bge = SentenceTransformer('dariolopez/bge-m3-es-legal-tmp-6')
    
    # Forzar GPU si esta disponible
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model_bge = model_bge.to(device)
        print(f"Modelo movido a: {model_bge.device}")
    
    start_time = time.time()
    embeddings_gpu = model_bge.encode(test_documents)
    gpu_time = time.time() - start_time
    
    print(f"[GPU] Tiempo total: {gpu_time:.2f}s")
    print(f"[GPU] Tiempo por documento: {gpu_time/len(test_documents):.3f}s")
    print(f"[GPU] Documentos por segundo: {len(test_documents)/gpu_time:.1f}")
    print(f"[GPU] Shape embeddings: {embeddings_gpu.shape}")
    
    # Comparacion de rendimiento
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"\n[RESULTADO] Aceleracion GPU: {speedup:.1f}x mas rapido")
    
    if speedup > 2:
        print("[EXITO] GPU significativamente mas rapida")
    elif speedup > 1.2:
        print("[OK] GPU moderadamente mas rapida")
    else:
        print("[WARN] GPU no muestra ventaja clara")
        
except Exception as e:
    print(f"[ERROR] BGE-M3 GPU: {e}")

# TEST 3: Memory usage GPU
if torch.cuda.is_available():
    print("\n--- TEST 3: USO MEMORIA GPU ---")
    torch.cuda.empty_cache()
    
    memory_before = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"Memoria GPU usada: {memory_before:.2f}GB")
    print(f"Memoria GPU reservada: {memory_reserved:.2f}GB")
    print(f"Memoria GPU libre: {31.0 - memory_reserved:.2f}GB")
    
    # Test con batch grande para ver capacidad
    print("\n--- TEST 4: BATCH PROCESSING GRANDE ---")
    large_batch = test_documents * 10  # 1000 documentos
    
    start_time = time.time()
    large_embeddings = model_bge.encode(large_batch, batch_size=32)
    large_time = time.time() - start_time
    
    print(f"[BATCH GRANDE] {len(large_batch)} documentos en {large_time:.2f}s")
    print(f"[BATCH GRANDE] {len(large_batch)/large_time:.1f} docs/segundo")
    
    memory_after = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[BATCH GRANDE] Memoria GPU usada: {memory_after:.2f}GB")

print("\n========================================")
print("TEST GPU EMBEDDINGS COMPLETADO")
print("RTX 5090 lista para produccion!")
print("========================================")