import torch
from transformers import AutoModel
import os

# Load WavLM Base model
print("Loading WavLM Base model...")
model_name = "microsoft/wavlm-base"
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
model.eval()

# Export to ONNX
print("Exporting to ONNX...")
dummy_input = torch.randn(1, 16000)
onnx_path = "/app/output/wavlm_base.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path,
    export_params=True,
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 1: "sequence"}}
)

print(f"Model exported to {onnx_path}")
print("File size:", os.path.getsize(onnx_path) / (1024 * 1024), "MB")
