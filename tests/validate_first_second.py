#!/usr/bin/env python3
"""
Generate validation features for FIRST SECOND of audio only.
This matches what JavaScript can process without stack overflow.
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'prep'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Speech-Articulatory-Coding'))

from validate_features import load_and_normalize_audio, extract_features_full

def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else '../Speech-Articulatory-Coding/sample_audio/sample1.wav'
    output_path = '../python_features_1sec.json'
    
    # Load audio
    print(f"Loading audio: {audio_path}")
    audio = load_and_normalize_audio(audio_path, sr=16000, normalize=False)
    sr = 16000
    
    # Take ONLY first second (16000 samples)
    audio_1sec = audio[:16000]
    print(f"  Original: {len(audio)} samples ({len(audio)/16000:.2f}s)")
    print(f"  Using first 1 second: {len(audio_1sec)} samples")
    
    # Save temporary file
    import soundfile as sf
    temp_audio = '/tmp/test_1sec.wav'
    sf.write(temp_audio, audio_1sec, sr)
    
    # Extract features
    linear_model_path = '../prep/convert_linear_model_pkl2json/wavlm_large-9_cut-10_mngu_linear.pkl'
    features = extract_features_full(temp_audio, linear_model_path=linear_model_path)
    
    # Get middle frame (batch dimension already removed by extract_features_full)
    ema = features['ema']  # Shape: (time, features) = (49, 12)
    print(f"EMA shape: {ema.shape}")
    
    middle_idx = ema.shape[0] // 2
    middle_frame = ema[middle_idx]  # (12,)
    
    print(f"\nExtracted {ema.shape[0]} frames with {ema.shape[1]} features each")
    print(f"Using middle frame #{middle_idx}")
    print(f"Middle frame: {middle_frame}")
    
    # Save
    output_data = {
        'audio': {
            'path': audio_path,
            'samples': 16000,
            'duration': 1.0
        },
        'ema': {
            'middle_frame': middle_frame.tolist(),
            'frame_index': middle_idx,
            'total_frames': int(ema.shape[0])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Saved to: {output_path}")

if __name__ == '__main__':
    main()

