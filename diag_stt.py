"""
Full STT Diagnostic Script v2 — with CUDA DLL fix and device selection.
"""

import os, sys, wave, time, math
import numpy as np
import sounddevice as sd

# ── CUDA DLL PATH FIX (same as stt_engine.py) ──────────────────────────────
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

print("=" * 70)
print("  FULL AUDIO / STT DIAGNOSTIC v2")
print("=" * 70)
print()

# ── Step 1: List input devices ──────────────────────────────────────────────
print("STEP 1: INPUT DEVICES")
print("-" * 70)
input_devices = []
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        is_default = i == sd.default.device[0]
        mark = " ← DEFAULT" if is_default else ""
        print(
            f"  [{i:>2}] {d['name'][:50]:<50} {d['default_samplerate']:>5.0f}Hz {d['max_input_channels']}ch{mark}"
        )
        input_devices.append((i, d))
print()

default_idx = sd.default.device[0]
default_name = sd.query_devices(default_idx)["name"]
print(f"  Default device: [{default_idx}] {default_name}")
print()

# ── Step 2: Record with DEFAULT device ──────────────────────────────────────
SR = 16000
DURATION = 5

print(f"STEP 2: RECORDING {DURATION}s with DEFAULT mic [{default_idx}]")
print("  Say 'HEY JARVIS OPEN CHROME' clearly after the beep...")
time.sleep(1)
print("  🔴 RECORDING NOW...")
raw = sd.rec(
    int(DURATION * SR), samplerate=SR, channels=1, dtype="int16", device=default_idx
)
sd.wait()
raw = raw.flatten()
print("  ✅ Done")
print()

# ── Step 3: Audio analysis ──────────────────────────────────────────────────
print("STEP 3: RAW AUDIO ANALYSIS")
print("-" * 70)
rms = np.sqrt(np.mean(raw.astype(float) ** 2))
peak = int(np.max(np.abs(raw)))

# Per-frame analysis
frame_size = 512
frame_rms = []
for i in range(0, len(raw) - frame_size, frame_size):
    chunk = raw[i : i + frame_size]
    frame_rms.append(np.sqrt(np.mean(chunk.astype(float) ** 2)))

print(f"  Overall RMS:  {rms:.1f} / 32768  ({rms/327.68:.2f}%)")
print(f"  Peak:         {peak} / 32768  ({peak/327.68:.2f}%)")
print(
    f"  Frame RMS:    min={min(frame_rms):.1f}  max={max(frame_rms):.1f}  avg={np.mean(frame_rms):.1f}"
)

if peak < 100:
    print("  ❌ CRITICALLY LOW — mic is essentially silent")
elif peak < 1000:
    print("  ⚠️  Very low mic level")
elif peak < 5000:
    print("  ⚠️  Low but usable mic level")
else:
    print("  ✅ Mic level OK")
print()


# ── Step 4: Save WAV files ──────────────────────────────────────────────────
def save_wav(filename, data):
    with wave.open(filename, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(data.tobytes())


save_wav("diag_raw.wav", raw)
print(f"  Saved diag_raw.wav (play this to hear what mic captures)")

# Peak-normalize to 80%
if peak > 0:
    scale = int(32768 * 0.8) / peak
    normalized = np.clip(raw.astype(np.float32) * scale, -32768, 32767).astype(np.int16)
    save_wav("diag_normalized.wav", normalized)
    norm_rms = np.sqrt(np.mean(normalized.astype(float) ** 2))
    print(
        f"  Saved diag_normalized.wav (peak-normalized, scale={scale:.1f}x, RMS={norm_rms:.0f})"
    )
print()

# ── Step 5: Load Whisper model ──────────────────────────────────────────────
print("STEP 5: LOADING WHISPER MODEL")
print("-" * 70)
from faster_whisper import WhisperModel

try:
    model = WhisperModel("small.en", device="cuda", compute_type="int8_float16")
    print("  ✅ Loaded small.en on CUDA")
except Exception as e:
    print(f"  CUDA failed ({e}), trying CPU...")
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    print("  ✅ Loaded small.en on CPU")
print()

# ── Step 6: Transcribe at different gains ─────────────────────────────────
print("STEP 6: TRANSCRIPTION AT DIFFERENT GAINS")
print("-" * 70)
print(f"  {'Gain':>6} | {'RMS':>6} | {'Conf':>5} | {'Text':<50}")
print("  " + "-" * 72)

best_gain = 1
best_conf = 0
best_text = ""

for g in [1, 5, 10, 25, 50, 100, 200, 500]:
    amp = np.clip(raw.astype(np.float32) * g, -32768, 32767).astype(np.int16)
    amp_rms = np.sqrt(np.mean(amp.astype(float) ** 2))
    audio_f = amp.astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio_f, beam_size=5, language="en", vad_filter=False
    )

    text = ""
    total_prob = 0.0
    n = 0
    for seg in segments:
        text += seg.text
        total_prob += seg.avg_logprob
        n += 1

    conf = math.exp(total_prob / n) if n > 0 else 0.0
    text = text.strip()

    # Is it a hallucination?
    is_hall = text.lower().rstrip(".!?,") in {
        "thank you",
        "thanks for watching",
        "thank you very much",
        "thanks for listening",
        "see you next time",
        "subscribe",
        "like and subscribe",
        "bye",
        "goodbye",
        "you",
        "oh",
        "",
    }

    marker = "❌ HALL" if is_hall else ""
    if not is_hall and conf > best_conf:
        best_gain = g
        best_conf = conf
        best_text = text

    display = text[:50] if text else "(empty)"
    print(f"  {g:>5}x | {amp_rms:>5.0f} | {conf:.3f} | {display} {marker}")

print()

# ── Step 7: Peak-normalized transcription ─────────────────────────────────
if peak > 0:
    print("STEP 7: PEAK-NORMALIZED TRANSCRIPTION")
    print("-" * 70)
    audio_f = normalized.astype(np.float32) / 32768.0
    segments, info = model.transcribe(
        audio_f, beam_size=5, language="en", vad_filter=False
    )
    text = ""
    total_prob = 0.0
    n = 0
    for seg in segments:
        text += seg.text
        total_prob += seg.avg_logprob
        n += 1
    conf = math.exp(total_prob / n) if n > 0 else 0.0
    text = text.strip()
    print(f"  Normalized (scale={scale:.0f}x): '{text}' (conf={conf:.3f})")

    is_correct = any(w in text.lower() for w in ["jarvis", "chrome", "open", "spotify"])
    if is_correct:
        print("  ✅ CORRECT — peak normalization works!")
        print(
            f"  → RECOMMENDED: Use peak normalization (dynamic gain) instead of fixed gain"
        )
    else:
        is_hall = text.lower().rstrip(".!?,") in {
            "thank you",
            "thanks for watching",
            "thank you very much",
            "",
        }
        if is_hall:
            print("  ❌ Still hallucinating — audio quality too poor")
        else:
            print(f"  ⚠️  Non-hallucination but unclear if correct")
    print()

# ── Step 8: Summary ──────────────────────────────────────────────────────
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Default mic:  [{default_idx}] {default_name}")
print(f"  Raw peak:     {peak} / 32768 ({peak/327.68:.2f}%)")
print(f"  Raw RMS:      {rms:.1f}")
if best_text and best_conf > 0:
    print(f"  Best result:  gain={best_gain}x → '{best_text}' (conf={best_conf:.3f})")
else:
    print(f"  Best result:  No valid transcription at any gain level")
print()
if peak < 100:
    print("  VERDICT: Mic captures almost NO audio. Software cannot fix this.")
    print("  RECOMMENDED: Switch to a USB microphone or adjust Windows mic settings:")
    print("    1. Right-click speaker icon → Sound Settings → Input")
    print("    2. Select your microphone → Properties")
    print("    3. Set volume to 100% and enable +30dB boost if available")
elif best_conf > 0.3 and any(
    w in best_text.lower() for w in ["jarvis", "chrome", "open"]
):
    print(f"  VERDICT: Working at gain={best_gain}x!")
else:
    print("  VERDICT: Mic level too low for reliable STT")
print("=" * 70)
