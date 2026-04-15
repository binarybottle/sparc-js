#!/usr/bin/env python3
"""Extract per-vowel z-scored EMA values using the ONNX pipeline.

Replicates the exact JS worker pipeline:
  1. Z-score normalize audio (full-file, matching Python SPARC convention)
  2. Zero-pad 160 samples each side
  3. Run WavLM ONNX (layer 9) -> hidden states [1, seq, 1024]
  4. Select second-to-last frame
  5. Linear projection -> 12 EMA channels (z-scored)

Channel order: TDX, TDY, TBX, TBY, TTX, TTY, LIX, LIY, ULX, ULY, LLX, LLY
"""
import json
import os
import wave
import struct

import numpy as np
import onnxruntime as ort

VOWEL_DIR = os.path.join(os.path.dirname(__file__), "vowels")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

VOWEL_MAP = {
    "ee": "i",
    "eh": "e",
    "ah": "a",
    "oh": "o",
    "oo": "u",
}

ART_KEYS = ["td", "tb", "tt", "li", "ul", "ll"]
CHANNEL_NAMES = ["td_x", "td_y", "tb_x", "tb_y", "tt_x", "tt_y",
                 "li_x", "li_y", "ul_x", "ul_y", "ll_x", "ll_y"]


def load_wav_16k(path):
    """Read a 16 kHz mono WAV file and return float32 samples."""
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono audio"
        assert wf.getframerate() == 16000, "Expected 16 kHz sample rate"
        n = wf.getnframes()
        raw = wf.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def zscore_normalize(audio):
    """Full-file z-score normalization (matching Python SPARC)."""
    mean = audio.mean()
    std = audio.std()
    if std < 1e-8:
        return audio
    return (audio - mean) / std


def run_wavlm_onnx(audio_norm, session):
    """Run WavLM ONNX model on z-scored, zero-padded audio.

    Returns hidden states tensor of shape (1, seq_len, 1024).
    """
    pad = 160
    padded = np.concatenate([np.zeros(pad, dtype=np.float32),
                             audio_norm,
                             np.zeros(pad, dtype=np.float32)])
    inp = padded.reshape(1, -1).astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: inp})
    return out[0]  # (1, seq_len, 1024)


def apply_linear_model(hidden_states, weights, biases):
    """Project hidden states to 12 EMA channels.

    Uses the second-to-last frame (matching JS worker).
    Returns dict {td: {x,y}, tb: {x,y}, ...} for each frame.
    """
    seq_len = hidden_states.shape[1]
    results_all_frames = []

    for t in range(seq_len):
        frame = hidden_states[0, t, :]  # (1024,)
        ema = biases + frame @ weights.T  # (12,)
        results_all_frames.append(ema)

    ema_all = np.stack(results_all_frames)  # (seq_len, 12)
    return ema_all


def ema_to_dict(ema_vec):
    """Convert 12-element EMA vector to articulator dict."""
    d = {}
    for i, key in enumerate(ART_KEYS):
        d[key] = {"x": round(float(ema_vec[i * 2]), 4),
                  "y": round(float(ema_vec[i * 2 + 1]), 4)}
    return d


def main():
    onnx_path = os.path.join(MODELS_DIR, "wavlm_large_layer9.onnx")
    lm_path = os.path.join(MODELS_DIR, "wavlm_linear_model.json")

    print(f"Loading WavLM ONNX model from {onnx_path} ...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print(f"Loading linear model from {lm_path} ...")
    with open(lm_path) as f:
        lm = json.load(f)
    weights = np.array(lm["weights"], dtype=np.float32)  # (12, 1024)
    biases = np.array(lm["biases"], dtype=np.float32)     # (12,)
    print(f"  weights shape: {weights.shape}, biases shape: {biases.shape}")

    results = {}

    for filename in sorted(VOWEL_MAP.keys()):
        vowel_id = VOWEL_MAP[filename]
        wav_path = os.path.join(VOWEL_DIR, f"{filename}.wav")

        if not os.path.exists(wav_path):
            print(f"  SKIP {wav_path} (not found)")
            continue

        audio = load_wav_16k(wav_path)
        audio_norm = zscore_normalize(audio)

        print(f"\n/{vowel_id}/ ({filename}.wav): {len(audio)} samples, "
              f"{len(audio)/16000:.2f}s")

        hidden = run_wavlm_onnx(audio_norm, sess)
        print(f"  WavLM output: {hidden.shape}")

        ema_all = apply_linear_model(hidden, weights, biases)
        print(f"  EMA frames: {ema_all.shape}")

        # Use all frames (not just second-to-last) to get a stable average
        means = ema_all.mean(axis=0)
        stds = ema_all.std(axis=0)

        print(f"  {'Channel':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        for i, name in enumerate(CHANNEL_NAMES):
            print(f"  {name:<8} {means[i]:8.4f} {stds[i]:8.4f} "
                  f"{ema_all[:, i].min():8.4f} {ema_all[:, i].max():8.4f}")

        results[vowel_id] = ema_to_dict(means)

    print("\n" + "=" * 60)
    print("VOWEL_Z_SCORES for visualization.js:")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    out_path = os.path.join(os.path.dirname(__file__), "vowel_zscores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
