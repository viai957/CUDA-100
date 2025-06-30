import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim
from whisper_cuda_model import convert_from_original_whisper

def load_audio(audio_path, device):
    """Load and preprocess audio for Whisper inference"""
    # Use whisper's audio loading function
    mel = whisper.log_mel_spectrogram(audio_path).to(device)
    return mel.unsqueeze(0)  # Add batch dimension

def benchmark_inference_time(model, mel, num_runs=10):
    """Benchmark inference time for a given model"""
    # Warmup run
    with torch.no_grad():
        _ = model.encoder(mel)
    
    # Benchmark encoding
    torch.cuda.synchronize()
    encode_times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            encoded = model.encoder(mel)
        torch.cuda.synchronize()
        encode_times.append(time.time() - start)
    
    # Prepare for decoding
    tokens = torch.tensor([[1, 1]]).to(mel.device)  # Start tokens
    
    # Warmup for decoding
    with torch.no_grad():
        _ = model.decoder(tokens, encoded)
    
    # Benchmark decoding (without KV caching)
    torch.cuda.synchronize()
    decode_times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model.decoder(tokens, encoded)
        torch.cuda.synchronize()
        decode_times.append(time.time() - start)
    
    # Benchmark KV caching
    kv_cache = {}
    model.install_kv_cache_hooks(kv_cache)
    
    # Warmup with KV cache
    with torch.no_grad():
        _ = model.decoder(tokens, encoded)
    
    # Benchmark decoding (with KV caching)
    torch.cuda.synchronize()
    kv_cache_times = []
    for i in range(num_runs):
        tokens = torch.tensor([[1, 1 + i]]).to(mel.device)  # Vary tokens slightly
        start = time.time()
        with torch.no_grad():
            _ = model.decoder(tokens, encoded)
        torch.cuda.synchronize()
        kv_cache_times.append(time.time() - start)
    
    return {
        "encode": encode_times,
        "decode": decode_times,
        "kv_cache": kv_cache_times,
    }

def plot_results(standard_times, cuda_times, output_path="benchmark_results.png"):
    """Plot benchmark results"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, phase in enumerate(["encode", "decode", "kv_cache"]):
        axs[i].boxplot([standard_times[phase], cuda_times[phase]], labels=["Standard", "CUDA"])
        axs[i].set_title(f"{phase.capitalize()} Time")
        axs[i].set_ylabel("Time (seconds)")
    
    fig.suptitle("Whisper Performance Benchmarks")
    plt.tight_layout()
    
    # Calculate speedup
    encode_speedup = np.mean(standard_times["encode"]) / np.mean(cuda_times["encode"])
    decode_speedup = np.mean(standard_times["decode"]) / np.mean(cuda_times["decode"])
    kv_speedup = np.mean(standard_times["kv_cache"]) / np.mean(cuda_times["kv_cache"])
    
    print(f"Encode speedup: {encode_speedup:.2f}x")
    print(f"Decode speedup: {decode_speedup:.2f}x")
    print(f"KV cache speedup: {kv_speedup:.2f}x")
    
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")

def run_transcription_benchmark(standard_model, cuda_model, audio_path, num_runs=3):
    """Benchmark full transcription with both models"""
    # Load audio
    audio = whisper.load_audio(audio_path)
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
    
    # Benchmark standard model
    torch.cuda.synchronize()
    standard_times = []
    for _ in tqdm(range(num_runs), desc="Standard Whisper"):
        start = time.time()
        _ = whisper.transcribe(standard_model, audio_path)
        torch.cuda.synchronize()
        standard_times.append(time.time() - start)
    
    # Benchmark CUDA model
    torch.cuda.synchronize()
    cuda_times = []
    for _ in tqdm(range(num_runs), desc="CUDA Whisper"):
        start = time.time()
        # We need to use the original transcribe with our model
        _ = whisper.transcribe(cuda_model, audio_path)
        torch.cuda.synchronize()
        cuda_times.append(time.time() - start)
    
    print(f"\nAverage transcription time (Standard): {np.mean(standard_times):.2f}s")
    print(f"Average transcription time (CUDA): {np.mean(cuda_times):.2f}s")
    print(f"Transcription speedup: {np.mean(standard_times) / np.mean(cuda_times):.2f}x")
    
    return standard_times, cuda_times

def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper vs CUDA-optimized Whisper")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file for transcription benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default="benchmark_results.png", help="Output path for benchmark results")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {args.model} Whisper model...")
    standard_model = whisper.load_model(args.model).to(device)
    
    print("Converting to CUDA-optimized model...")
    cuda_model = convert_from_original_whisper(standard_model)
    
    # Generate random mel spectrogram if no audio file is provided
    if args.audio is None:
        print("Generating random mel spectrogram...")
        mel = torch.randn(1, 80, 3000).to(device).half()  # Typical mel spectrogram shape
    else:
        print(f"Loading audio from {args.audio}...")
        mel = load_audio(args.audio, device)
    
    print(f"Running component benchmark with {args.runs} runs...")
    standard_times = benchmark_inference_time(standard_model, mel, args.runs)
    cuda_times = benchmark_inference_time(cuda_model, mel, args.runs)
    
    plot_results(standard_times, cuda_times, args.output)
    
    # Run transcription benchmark if audio file is provided
    if args.audio:
        print("\nRunning transcription benchmark...")
        run_transcription_benchmark(standard_model, cuda_model, args.audio, num_runs=3)

if __name__ == "__main__":
    main() 