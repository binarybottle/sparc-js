"""
Convert WavLM PyTorch model to ONNX, truncated to 9 layers.

Produces two files:
  - wavlm_large_layer9.onnx       (FP32, ~483 MB)
  - quantized_wavlm_large_layer9.onnx (QUInt8, ~122 MB)

Requirements:
  pip install torch transformers onnx onnxruntime
"""

import torch
import onnx
import os
from transformers import WavLMModel

wavlm_model_version = "microsoft/wavlm-large"
output_file = "wavlm_large_layer9.onnx"

print("Loading WavLM model...")
model = WavLMModel.from_pretrained(wavlm_model_version)


class WavLMLayer9(torch.nn.Module):
    def __init__(self, wavlm_model):
        super().__init__()
        self.wavlm = wavlm_model

    def forward(self, input_values):
        outputs = self.wavlm(input_values, output_hidden_states=True)
        return outputs.hidden_states[9]


print("Creating truncated model (layers 0-9)...")
truncated_model = WavLMLayer9(model)
truncated_model.eval()

print("Exporting to ONNX (opset 14)...")
dummy_input = torch.randn(1, 16000)
torch.onnx.export(
    truncated_model,
    dummy_input,
    output_file,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "audio_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=14,
    do_constant_folding=True,
    verbose=False,
)

print("Verifying ONNX model...")
onnx_model = onnx.load(output_file)
onnx.checker.check_model(onnx_model)

print(f"FP32 model: {output_file} ({os.path.getsize(output_file) / (1024 * 1024):.1f} MB)")

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_file = "quantized_" + output_file
    print("Quantizing model (QUInt8)...")
    quantize_dynamic(output_file, quantized_file, weight_type=QuantType.QUInt8)
    print(f"Quantized model: {quantized_file} ({os.path.getsize(quantized_file) / (1024 * 1024):.1f} MB)")
except ImportError:
    print("Skipping quantization (onnxruntime.quantization not available)")
