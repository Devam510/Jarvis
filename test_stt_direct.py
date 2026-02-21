"""Quick diagnostic: record 4 seconds, normalize, transcribe."""

import os
import sys
import time
import math
import numpy as np
import sounddevice as sd

# Setup CUDA DLL paths (same as stt_engine.py)
for site_dir in sys.path:
    nvidia_dir = os.path.join(site_dir, "nvidia")
    if not os.path.isdir(nvidia_dir):
        continue
    for subdir in os.listdir(nvidia_dir):
        for lib_dir in ("bin", "lib"):
            dll_path = os.path.join(nvidia_dir, subdir, lib_dir)
            if os.path.isdir(dll_path) and dll_path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = dll_path + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(dll_path)
                    except OSError:
                        pass

print("=" * 60)
print("SAY: 'Hey Jarvis, open Chrome' (recording 4 seconds)")
print("=" * 60)
time.sleep(0.5)

audio = sd.rec(int(4 * 16000), samplerate=16000, channels=1, dtype="int16")
sd.wait()
audio = audio.flatten()

raw_peak = int(np.max(np.abs(audio)))
raw_rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
print(f"\nRaw audio: peak={raw_peak}, RMS={raw_rms:.0f}")

# Peak-normalize the ENTIRE segment (not per-frame!)
TARGET = int(32768 * 0.8)
if raw_peak > 50:
    scale = TARGET / raw_peak
    normalized = np.clip(audio.astype(np.float32) * scale, -32768, 32767).astype(
        np.int16
    )
    norm_peak = int(np.max(np.abs(normalized)))
    norm_rms = float(np.sqrt(np.mean(normalized.astype(np.float32) ** 2)))
    print(f"Normalized: peak={norm_peak}, RMS={norm_rms:.0f}, scale={scale:.2f}x")
else:
    normalized = audio
    print("Audio too quiet to normalize")

# Convert to float32 [-1, 1]
audio_f32 = normalized.astype(np.float32) / 32768.0
f32_peak = float(np.max(np.abs(audio_f32)))
f32_rms = float(np.sqrt(np.mean(audio_f32**2)))
print(f"Float32: peak={f32_peak:.3f}, RMS={f32_rms:.4f}")

# Transcribe
from faster_whisper import WhisperModel

print("\nLoading Whisper model (small.en, CUDA)...")
try:
    model = WhisperModel("small.en", device="cuda", compute_type="int8_float16")
    print("Model loaded on CUDA")
except Exception as e:
    print(f"CUDA failed ({e}), trying CPU...")
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    print("Model loaded on CPU")

print("Transcribing...")
t0 = time.time()
segments, info = model.transcribe(
    audio_f32,
    beam_size=5,
    language="en",
    vad_filter=False,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
)

print(f"\nResults (latency={((time.time()-t0)*1000):.0f}ms):")
for seg in segments:
    conf = math.exp(seg.avg_logprob)
    print(f'  [{seg.start:.1f}-{seg.end:.1f}s] "{seg.text.strip()}" conf={conf:.3f}')

print("\nDone.")
