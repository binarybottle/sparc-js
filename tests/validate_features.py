#!/usr/bin/env python3
"""
Validate SPARC feature extraction between Python and JavaScript implementations.

This script:
1. Loads an audio file
2. Extracts features using the Python SPARC implementation
3. Saves the raw features for comparison with JavaScript
4. Provides ground truth for validation

Usage:
    python validate_features.py <audio_file> [output_json]

Requirements:
    pip install torch transformers scipy numpy soundfile
"""

import argparse
import json
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Speech-Articulatory-Coding'))

from scipy.signal import butter, filtfilt
import soundfile as sf


def butter_lowpass(cutoff, fs, order=5):
    """Create Butterworth lowpass filter coefficients."""
    return butter(order, cutoff, fs=fs, btype='low')


def butter_lowpass_filter(data, cutoff, fs, axis=1, order=5):
    """Apply Butterworth lowpass filter to data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y


def load_and_normalize_audio(audio_path, sr=16000, normalize=True):
    """Load audio file and normalize like SPARC does."""
    wav, orig_sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(wav.shape) > 1:
        wav = wav.mean(axis=-1)
    
    # Resample if needed
    if orig_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    
    # Normalize (z-score) - this is what SPARC does
    if normalize:
        wav = (wav - wav.mean()) / wav.std()
    
    return wav.astype(np.float32)


def extract_wavlm_features(wav, model, target_layer=9):
    """Extract WavLM hidden states at target layer."""
    import torch
    
    # Convert to tensor
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)
    
    # Create attention mask
    attention_mask = torch.ones_like(wav_tensor)
    
    with torch.no_grad():
        outputs = model(wav_tensor, attention_mask=attention_mask, output_hidden_states=True)
    
    # Get target layer hidden states
    hidden_states = outputs.hidden_states[target_layer]
    return hidden_states.cpu().numpy()


def apply_linear_model(hidden_states, linear_model_path):
    """Apply linear projection to get EMA features."""
    import pickle
    
    # Load linear model
    with open(linear_model_path, 'rb') as f:
        linear_model = pickle.load(f)
    
    # Reshape for linear model: [batch * seq_len, hidden_dim]
    batch_size, seq_len, hidden_dim = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden_dim)
    
    # Apply linear projection
    ema_flat = linear_model.predict(hidden_flat)
    
    # Reshape back: [batch, seq_len, 12]
    ema = ema_flat.reshape(batch_size, seq_len, 12)
    
    return ema


def extract_features_full(audio_path, 
                          wavlm_model_name='microsoft/wavlm-large',
                          target_layer=9,
                          freqcut=10,
                          ft_sr=50,
                          sr=16000,
                          linear_model_path=None):
    """
    Extract SPARC features using the full Python pipeline.
    
    Returns dict with:
        - ema: [seq_len, 12] articulation features
        - hidden_states: [seq_len, hidden_dim] raw WavLM features
        - audio: [samples] normalized audio
    """
    from transformers import WavLMModel
    
    print(f"Loading audio: {audio_path}")
    wav = load_and_normalize_audio(audio_path, sr=sr)
    print(f"  Samples: {len(wav)}, Duration: {len(wav)/sr:.2f}s")
    
    print(f"Loading WavLM model: {wavlm_model_name}")
    model = WavLMModel.from_pretrained(wavlm_model_name)
    
    # Truncate to target layer for efficiency
    model.encoder.layers = model.encoder.layers[:target_layer + 1]
    model.eval()
    
    print(f"Extracting hidden states at layer {target_layer}...")
    hidden_states = extract_wavlm_features(wav, model, target_layer)
    print(f"  Hidden states shape: {hidden_states.shape}")
    
    # Apply lowpass filter (this is what Python SPARC does!)
    if freqcut > 0:
        print(f"Applying {freqcut}Hz lowpass filter...")
        hidden_states = butter_lowpass_filter(hidden_states, freqcut, ft_sr, axis=1)
    
    # Apply linear model if provided
    ema = None
    if linear_model_path:
        print(f"Applying linear model: {linear_model_path}")
        ema = apply_linear_model(hidden_states, linear_model_path)
        print(f"  EMA shape: {ema.shape}")
    
    return {
        'audio': wav,
        'hidden_states': hidden_states[0],  # Remove batch dim
        'ema': ema[0] if ema is not None else None,
        'config': {
            'wavlm_model': wavlm_model_name,
            'target_layer': target_layer,
            'freqcut': freqcut,
            'ft_sr': ft_sr,
            'sr': sr,
            'hidden_dim': hidden_states.shape[-1]
        }
    }


def save_validation_data(features, output_path):
    """Save features in JSON format for JavaScript comparison."""
    data = {
        'config': features['config'],
        'audio_samples': len(features['audio']),
        'num_frames': features['hidden_states'].shape[0],
        'hidden_dim': features['hidden_states'].shape[1],
        # Sample of hidden states for validation
        'hidden_states_sample': {
            'first_frame': features['hidden_states'][0, :10].tolist(),
            'middle_frame': features['hidden_states'][len(features['hidden_states'])//2, :10].tolist(),
            'last_frame': features['hidden_states'][-1, :10].tolist()
        }
    }
    
    if features['ema'] is not None:
        data['ema_features'] = {
            'num_features': features['ema'].shape[1],
            'feature_names': ['td_x', 'td_y', 'tb_x', 'tb_y', 'tt_x', 'tt_y',
                             'li_x', 'li_y', 'ul_x', 'ul_y', 'll_x', 'll_y'],
            # Full EMA sequence for comparison
            'all_frames': features['ema'].tolist(),
            # Statistics
            'mean': features['ema'].mean(axis=0).tolist(),
            'std': features['ema'].std(axis=0).tolist(),
            'min': features['ema'].min(axis=0).tolist(),
            'max': features['ema'].max(axis=0).tolist()
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved validation data to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate SPARC feature extraction')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--output', '-o', default='python_features.json',
                       help='Output JSON file for validation')
    parser.add_argument('--linear-model', '-l', 
                       default='../Speech-Articulatory-Coding/wavlm_large-9_cut-10_mngu_linear.pkl',
                       help='Path to linear model pickle file')
    args = parser.parse_args()
    
    # Check if linear model exists, try alternate paths
    linear_model_path = None
    possible_paths = [
        args.linear_model,
        '../prep/convert_linear_model_pkl2json/wavlm_large-9_cut-10_mngu_linear.pkl',
        'prep/convert_linear_model_pkl2json/wavlm_large-9_cut-10_mngu_linear.pkl',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            linear_model_path = path
            break
    
    if linear_model_path:
        print(f"Using linear model: {linear_model_path}")
    else:
        print("Warning: Linear model not found, will only extract WavLM features")
    
    # Extract features
    features = extract_features_full(
        args.audio_file,
        linear_model_path=linear_model_path
    )
    
    # Save for validation
    save_validation_data(features, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Audio samples: {len(features['audio'])}")
    print(f"Hidden states: {features['hidden_states'].shape}")
    
    if features['ema'] is not None:
        ema = features['ema']
        print(f"EMA features: {ema.shape}")
        print("\nEMA Statistics (per articulator):")
        print("-" * 50)
        names = ['td_x', 'td_y', 'tb_x', 'tb_y', 'tt_x', 'tt_y',
                 'li_x', 'li_y', 'ul_x', 'ul_y', 'll_x', 'll_y']
        for i, name in enumerate(names):
            print(f"  {name:6s}: mean={ema[:,i].mean():7.4f}, std={ema[:,i].std():6.4f}, "
                  f"range=[{ema[:,i].min():7.4f}, {ema[:,i].max():7.4f}]")


if __name__ == '__main__':
    main()





