#!/bin/bash
# Quick validation test for SPARC feature extraction

echo "🧪 SPARC Feature Validation Test"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "../app.js" ]; then
    echo "❌ Error: Run this script from the tests/ directory"
    exit 1
fi

# Check if sample audio exists
SAMPLE_AUDIO="../Speech-Articulatory-Coding/sample_audio/sample1.wav"
if [ ! -f "$SAMPLE_AUDIO" ]; then
    echo "⚠️  Sample audio not found. Using any available audio file..."
    # Find first .wav file
    SAMPLE_AUDIO=$(find ../Speech-Articulatory-Coding -name "*.wav" -type f | head -1)
    if [ -z "$SAMPLE_AUDIO" ]; then
        echo "❌ No audio files found. Please provide a .wav file."
        echo ""
        echo "Usage: ./quick_validation.sh [path/to/audio.wav]"
        exit 1
    fi
fi

# Use provided audio file if given
if [ ! -z "$1" ]; then
    SAMPLE_AUDIO="$1"
fi

echo "📁 Using audio file: $SAMPLE_AUDIO"
echo ""

# Step 1: Generate Python ground truth
echo "Step 1: Generating Python ground truth..."
cd "$(dirname "$0")"

if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Install poetry or use pip directly:"
    echo "   pip install torch transformers scipy soundfile"
    exit 1
fi

# Check if in prep directory or tests directory
if [ -f "../prep/pyproject.toml" ]; then
    cd ../prep
    poetry run python ../tests/validate_features.py "$SAMPLE_AUDIO" -o ../tests/python_features.json
    cd ../tests
else
    poetry run python validate_features.py "$SAMPLE_AUDIO" -o python_features.json
fi

if [ $? -ne 0 ]; then
    echo "❌ Python feature extraction failed"
    exit 1
fi

echo "✅ Python features generated: python_features.json"
echo ""

# Step 2: Instructions for JavaScript validation
echo "Step 2: Validate JavaScript extraction"
echo "--------------------------------------"
echo ""
echo "1. Open validate_js_features.html in your browser:"
echo "   file://$(pwd)/validate_js_features.html"
echo ""
echo "2. Load the Python ground truth:"
echo "   - Click 'Load Python JSON'"
echo "   - Select: python_features.json"
echo ""
echo "3. Load the same audio file:"
echo "   - Click 'Load Audio'"
echo "   - Select: $SAMPLE_AUDIO"
echo ""
echo "4. Click 'Run Validation'"
echo ""
echo "✅ Expected results:"
echo "   - Correlation > 0.9 for all features"
echo "   - RMSE < 0.5 for all features"
echo "   - All features show ✅ (green checkmark)"
echo ""
echo "If validation passes, your JavaScript implementation matches Python! 🎉"

