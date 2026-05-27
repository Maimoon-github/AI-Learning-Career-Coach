#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Installing Ollama (if not already present)..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama already installed"
fi

echo "📦 Pulling required models..."
MODELS=("llama3.2:3b" "llama3.1:70b")

for model in "${MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        echo "✅ Model $model already present"
    else
        echo "⬇️  Pulling $model (this may take a while)..."
        ollama pull "$model"
        echo "✅ $model pulled successfully"
    fi
done

echo "🎉 All models ready:"
ollama list | grep -E "llama3.2:3b|llama3.1:70b"