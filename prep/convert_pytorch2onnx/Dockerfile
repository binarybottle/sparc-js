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
