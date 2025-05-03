# Create a directory for the conversion
mkdir wavlm_conversion
cd wavlm_conversion

# Create a Python script for conversion
cat > convert_wavlm.py << 'EOL'
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
EOL

# Create a Dockerfile
cat > Dockerfile << 'EOL'
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Install dependencies
RUN pip install transformers==4.18.0 onnx==1.11.0 protobuf==3.20.0

# Set up working directory
WORKDIR /app
COPY convert_wavlm.py .

# Create output directory
RUN mkdir -p /app/output

# Run the conversion script
CMD ["python", "convert_wavlm.py"]
EOL

# Build the Docker image
docker build -t wavlm-converter .

# Run the container with volume mapping
mkdir -p output
docker run -v $(pwd)/output:/app/output wavlm-converter

# Check the output
ls -la output/