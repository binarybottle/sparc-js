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

# Step 1: Generate Python ground truth (1 second)
echo "Step 1: Generating Python ground truth (first 1 second)..."
cd "$(dirname "$0")"

if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Install poetry or use pip directly:"
    echo "   pip install torch transformers scipy soundfile"
    exit 1
fi

# Generate 1-second validation data
if [ -f "../prep/pyproject.toml" ]; then
    cd ../prep
    poetry run python ../tests/validate_first_second.py "$SAMPLE_AUDIO"
    cd ../tests
else
    echo "❌ Error: prep/pyproject.toml not found"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "❌ Python feature extraction failed"
    exit 1
fi

echo "✅ Python features generated: python_features_1sec.json"
echo ""

# Step 2: Instructions for JavaScript validation
echo "Step 2: Validate JavaScript extraction"
echo "--------------------------------------"
echo ""
echo "1. Start local web server from project root:"
echo "   cd .. && python3 server.py"
echo ""
echo "2. Open validation page in your browser:"
echo "   http://localhost:8000/validation.html"
echo ""
echo "3. Load the Python ground truth:"
echo "   - Click 'Load Python JSON'"
echo "   - Select: tests/python_features_1sec.json"
echo ""
echo "4. Load the same audio file:"
echo "   - Click 'Load Audio File'"
echo "   - Select: Speech-Articulatory-Coding/sample_audio/sample1.wav"
echo ""
echo "5. Click 'Extract Features'"
echo ""
echo "✅ Expected results:"
echo "   - Average Difference < 0.3"
echo "   - Most features show ✅ Match or ⚠️ Close"
echo "   - Status: ⚠️ CLOSE or ✅ PASS"
echo ""
echo "If average difference < 0.3, your JavaScript implementation is accurate! 🎉"

