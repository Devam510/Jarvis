"""Test script to find optimal mic gain for wake word detection."""

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

print("=" * 50)
print("  MIC CALIBRATION TEST")
print("=" * 50)
print()

# Step 1: Measure ambient noise for 2 seconds
print("Step 1: Recording 2s of SILENCE (don't speak)...")
ambient = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="int16")
sd.wait()
ambient_flat = ambient.flatten()
ambient_rms = np.sqrt(np.mean(ambient_flat.astype(float) ** 2))
print(f"  Ambient RMS: {ambient_rms:.1f}")

# Step 2: Record the user saying "Hey Jarvis"
print()
print("Step 2: Say 'HEY JARVIS' clearly now! (recording 4 seconds)")
speech = sd.rec(int(4 * 16000), samplerate=16000, channels=1, dtype="int16")
sd.wait()
speech_flat = speech.flatten()
speech_rms = np.sqrt(np.mean(speech_flat.astype(float) ** 2))
speech_peak_rms = max(
    np.sqrt(np.mean(speech_flat[i : i + 512].astype(float) ** 2))
    for i in range(0, len(speech_flat) - 512, 512)
)
print(f"  Speech RMS: {speech_rms:.1f} (peak frame: {speech_peak_rms:.1f})")

# Step 3: Test different gains
print()
print("Step 3: Testing different gains against the model...")
print(f"{'Gain':>6} | {'Peak Score':>10} | {'False on Silence':>16} | Verdict")
print("-" * 60)

for gain in [1, 5, 10, 15, 20, 30, 50]:
    model.reset()

    # Test on speech
    amp_speech = np.clip(speech_flat.astype(float) * gain, -32768, 32767).astype(
        np.int16
    )
    max_score = 0.0
    for i in range(0, len(amp_speech) - 1280, 1280):
        chunk = amp_speech[i : i + 1280]
        p = model.predict(chunk)
        s = p.get("hey_jarvis", 0)
        if s > max_score:
            max_score = s

    model.reset()

    # Test on silence
    amp_ambient = np.clip(ambient_flat.astype(float) * gain, -32768, 32767).astype(
        np.int16
    )
    max_false = 0.0
    for i in range(0, len(amp_ambient) - 1280, 1280):
        chunk = amp_ambient[i : i + 1280]
        p = model.predict(chunk)
        s = p.get("hey_jarvis", 0)
        if s > max_false:
            max_false = s

    # Verdict
    if max_score > 0.5 and max_false < 0.3:
        verdict = "GOOD"
    elif max_score > 0.3 and max_false < 0.3:
        verdict = "OK (lower threshold needed)"
    elif max_false > 0.5:
        verdict = "BAD (false triggers!)"
    else:
        verdict = "WEAK (voice not detected)"

    print(f"{gain:>6} | {max_score:>10.4f} | {max_false:>16.4f} | {verdict}")

print()
print("Pick a gain where voice scores high and silence stays low.")
