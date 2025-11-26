#!/usr/bin/env python3
"""
Convert WavLM Large layer 9 to ONNX format for JavaScript SPARC implementation.

This script creates an ONNX model that matches the linear model's expected input:
- Model: microsoft/wavlm-large
- Target layer: 9
- Output dimensions: [batch, seq_len, 1024]

Usage:
    python convert_wavlm_large_to_onnx.py

Requirements:
    pip install torch transformers onnx onnxruntime
    pip install onnxruntime-extensions  # for quantization
"""

import torch
import onnx
import os
from transformers import WavLMModel

# Model configuration - matches SPARC Python implementation
WAVLM_MODEL = "microsoft/wavlm-large"  # NOT wavlm-base!
TARGET_LAYER = 9
OUTPUT_FILE = "wavlm_large_layer9.onnx"
QUANTIZED_OUTPUT_FILE = "wavlm_large_layer9_quantized.onnx"


class WavLMLayer9(torch.nn.Module):
    """
    Wrapper that extracts only layer 9 hidden states from WavLM.
    This matches the SPARC Python implementation in inversion.py.
    """
    def __init__(self, wavlm_model, target_layer=9):
        super().__init__()
        self.wavlm = wavlm_model
        self.target_layer = target_layer

    def forward(self, input_values):
        outputs = self.wavlm(input_values, output_hidden_states=True)
        # Extract target layer hidden states
        # hidden_states is a tuple: (embedding, layer0, layer1, ..., layerN)
        # Layer 9 is at index 9 (0-indexed layers after embedding)
        target_hidden = outputs.hidden_states[self.target_layer]
        return target_hidden


def main():
    print(f"Loading WavLM model: {WAVLM_MODEL}")
    print(f"Target layer: {TARGET_LAYER}")
    
    # Load the full WavLM Large model
    model = WavLMModel.from_pretrained(WAVLM_MODEL)
    
    # Print model info
    print(f"Model hidden size: {model.config.hidden_size}")  # Should be 1024
    print(f"Number of layers: {model.config.num_hidden_layers}")  # Should be 12 or 24
    
    # Create the truncated model
    print("Creating truncated model (layer 9 only)...")
    truncated_model = WavLMLayer9(model, target_layer=TARGET_LAYER)
    truncated_model.eval()
    
    # Verify output dimensions with a test input
    print("Verifying model output dimensions...")
    with torch.no_grad():
        test_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz
        test_output = truncated_model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {test_output.shape}")  # Should be [1, seq_len, 1024]
        
        if test_output.shape[-1] != 1024:
            raise ValueError(f"Expected hidden size 1024, got {test_output.shape[-1]}")
    
    # Export to ONNX
    print(f"Exporting to ONNX: {OUTPUT_FILE}")
    dummy_input = torch.randn(1, 16000)
    
    # Use legacy ONNX export (dynamo=False) for compatibility with Transformers
    # The new exporter (dynamo=True) has issues with WavLM attention mechanisms
    torch.onnx.export(
        truncated_model,
        dummy_input,
        OUTPUT_FILE,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "audio_length"},
            "output": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamo=False  # Use legacy exporter
    )
    
    # Verify the exported model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(OUTPUT_FILE)
    onnx.checker.check_model(onnx_model)
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ Model exported successfully: {OUTPUT_FILE}")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Test with ONNX Runtime
    print("Testing with ONNX Runtime...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(OUTPUT_FILE)
    ort_input = {session.get_inputs()[0].name: test_input.numpy()}
    ort_output = session.run(None, ort_input)[0]
    
    print(f"ONNX Runtime output shape: {ort_output.shape}")
    
    # Compare with PyTorch output
    with torch.no_grad():
        pytorch_output = truncated_model(test_input).numpy()
    
    max_diff = abs(ort_output - pytorch_output).max()
    print(f"Max difference between PyTorch and ONNX: {max_diff:.6f}")
    
    if max_diff > 0.01:
        print("⚠️  Warning: Significant difference between PyTorch and ONNX outputs!")
    else:
        print("✅ ONNX output matches PyTorch output")
    
    # Quantize the model
    print("\nQuantizing model...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            OUTPUT_FILE,
            QUANTIZED_OUTPUT_FILE,
            weight_type=QuantType.QUInt8
        )
        
        quantized_size_mb = os.path.getsize(QUANTIZED_OUTPUT_FILE) / (1024 * 1024)
        print(f"✅ Quantized model: {QUANTIZED_OUTPUT_FILE}")
        print(f"   File size: {quantized_size_mb:.2f} MB ({100 * quantized_size_mb / file_size_mb:.1f}% of original)")
        
        # Test quantized model
        print("Testing quantized model...")
        quant_session = ort.InferenceSession(QUANTIZED_OUTPUT_FILE)
        quant_output = quant_session.run(None, ort_input)[0]
        quant_diff = abs(quant_output - pytorch_output).max()
        print(f"Max difference (quantized): {quant_diff:.6f}")
        
    except ImportError:
        print("⚠️  Skipping quantization - onnxruntime.quantization not available")
        print("   Install with: pip install onnxruntime-extensions")
    
    print("\n" + "=" * 60)
    print("IMPORTANT: After generating the ONNX model, copy it to:")
    print("  sparc-js/models/wavlm_large_layer9_quantized.onnx")
    print("\nThen update sparc-worker.js to use the new model path.")
    print("=" * 60)


if __name__ == "__main__":
    main()

