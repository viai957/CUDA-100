#!/usr/bin/env python3
import argparse
import time
import torch
import sys

import whisper
from whisper_cuda_model import convert_from_original_whisper

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using CUDA-optimized Whisper")
    parser.add_argument("audio", type=str, help="Path to audio file to transcribe")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--use-standard", action="store_true", help="Use standard Whisper model instead of CUDA-optimized")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g., 'en', 'fr', 'de')")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Task to perform")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("Warning: No GPU detected. Running on CPU will be slow and won't benefit from CUDA optimizations.")
        if not args.use_standard:
            print("Forcing --use-standard since no GPU is available.")
            args.use_standard = True
    
    print(f"Loading {args.model} Whisper model...")
    standard_model = whisper.load_model(args.model).to(device)
    
    if args.use_standard:
        model = standard_model
        print("Using standard Whisper model")
    else:
        print("Converting to CUDA-optimized model...")
        model = convert_from_original_whisper(standard_model)
        print("Using CUDA-optimized Whisper model")
    
    print(f"Transcribing {args.audio}...")
    start_time = time.time()
    
    # Set transcription options
    options = {
        "task": args.task,
        "verbose": True,
    }
    if args.language:
        options["language"] = args.language
    
    # Run transcription
    result = whisper.transcribe(model, args.audio, **options)
    
    elapsed = time.time() - start_time
    print(f"\nTranscription completed in {elapsed:.2f} seconds")
    
    # Print transcription
    print("\nTranscription:")
    print("=" * 80)
    if "text" in result:
        print(result["text"])
    else:
        print("No transcription generated")
    
    print("\nSegments:")
    print("-" * 80)
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.2f}s -> {end:.2f}s] {text}")

if __name__ == "__main__":
    main() 