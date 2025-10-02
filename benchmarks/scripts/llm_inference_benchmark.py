#!/usr/bin/env python3
"""
LLM Inference Benchmark Script
Measures throughput, latency, and resource usage for LLM inference
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any

import torch
import numpy as np

def setup_utf8_encoding():
    """Setup UTF-8 encoding for Windows compatibility"""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

setup_utf8_encoding()


class LLMBenchmark:
    """Benchmark harness for LLM inference"""

    def __init__(self, model_name: str, device: str = "cuda:0",
                 quantization: str = None):
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Quantization: {self.quantization or 'None (FP16)'}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model with optional quantization
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": self.device,
        }

        if self.quantization == "8bit":
            load_kwargs["load_in_8bit"] = True
        elif self.quantization == "4bit":
            load_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )

        print(f"Model loaded successfully on {self.model.device}")

    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {}

        memory_stats = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_stats[f"gpu_{i}_allocated_gb"] = round(allocated, 2)
            memory_stats[f"gpu_{i}_reserved_gb"] = round(reserved, 2)

        return memory_stats

    def run_inference(self, prompt: str, max_new_tokens: int = 100) -> Dict[str, Any]:
        """Run single inference and measure metrics"""

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # Measure time to first token and total time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for benchmarking
                pad_token_id=self.tokenizer.eos_token_id,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()

        total_time = end_time - start_time
        output_length = outputs.shape[1] - input_length

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        return {
            "input_tokens": input_length,
            "output_tokens": output_length,
            "total_tokens": outputs.shape[1],
            "total_time_s": round(total_time, 3),
            "tokens_per_second": round(output_length / total_time, 2),
            "generated_text": generated_text[:100] + "...",  # First 100 chars
        }

    def benchmark(self, prompts: List[str], max_new_tokens: int = 100,
                  warmup_runs: int = 3, test_runs: int = 10) -> Dict[str, Any]:
        """Run full benchmark with warmup and multiple test runs"""

        print(f"\nStarting benchmark:")
        print(f"  Warmup runs: {warmup_runs}")
        print(f"  Test runs: {test_runs}")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Max new tokens: {max_new_tokens}")

        # Warmup
        print(f"\nWarming up ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            prompt = prompts[i % len(prompts)]
            self.run_inference(prompt, max_new_tokens)
            print(f"  Warmup {i+1}/{warmup_runs} complete")

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test runs
        print(f"\nRunning benchmark ({test_runs} runs)...")
        results = []

        for i in range(test_runs):
            prompt = prompts[i % len(prompts)]
            result = self.run_inference(prompt, max_new_tokens)
            results.append(result)
            print(f"  Run {i+1}/{test_runs}: {result['tokens_per_second']} tokens/s")

        # Aggregate statistics
        tokens_per_second = [r["tokens_per_second"] for r in results]
        total_times = [r["total_time_s"] for r in results]

        stats = {
            "throughput_mean": round(np.mean(tokens_per_second), 2),
            "throughput_std": round(np.std(tokens_per_second), 2),
            "throughput_min": round(np.min(tokens_per_second), 2),
            "throughput_max": round(np.max(tokens_per_second), 2),
            "latency_mean_s": round(np.mean(total_times), 3),
            "latency_std_s": round(np.std(total_times), 3),
        }

        return {
            "stats": stats,
            "individual_runs": results,
            "gpu_memory": self.get_gpu_memory(),
        }


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument("--model", type=str, required=True,
                       help="HuggingFace model name or path")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on (cuda:0, cuda:1, cpu)")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", None],
                       default=None, help="Quantization type")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum new tokens to generate")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of test runs")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")

    args = parser.parse_args()

    # Sample prompts for testing
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the key differences between Python and JavaScript?",
        "Describe the process of photosynthesis.",
        "How does a neural network work?",
    ]

    print("="*60)
    print("LLM INFERENCE BENCHMARK")
    print("="*60)

    # System info
    print(f"\nSystem Information:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {props.total_memory // 1024**3}GB")

    # Run benchmark
    benchmark = LLMBenchmark(args.model, args.device, args.quantization)
    benchmark.load_model()

    results = benchmark.benchmark(
        prompts,
        max_new_tokens=args.max_tokens,
        warmup_runs=args.warmup,
        test_runs=args.runs
    )

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"\nThroughput:")
    print(f"  Mean: {results['stats']['throughput_mean']} tokens/s")
    print(f"  Std:  {results['stats']['throughput_std']} tokens/s")
    print(f"  Min:  {results['stats']['throughput_min']} tokens/s")
    print(f"  Max:  {results['stats']['throughput_max']} tokens/s")

    print(f"\nLatency:")
    print(f"  Mean: {results['stats']['latency_mean_s']}s")
    print(f"  Std:  {results['stats']['latency_std_s']}s")

    print(f"\nGPU Memory:")
    for key, value in results['gpu_memory'].items():
        print(f"  {key}: {value}GB")

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "quantization": args.quantization,
        "max_new_tokens": args.max_tokens,
        "warmup_runs": args.warmup,
        "test_runs": args.runs,
        "results": results,
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1]
        output_path = f"benchmarks/results/raw/{timestamp}_{model_short}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
