#!/bin/bash
# QuantAgent Setup Script
# Installs all Python dependencies via pip (no conda required).

set -e

echo "=== QuantAgent Setup ==="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python ${PYTHON_VERSION}"

# Warn about API keys
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: No API keys set. For the agent pipeline, set one of:"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
fi

# Install dependencies with pip
echo "Installing Python dependencies with pip..."
python3 -m pip install --upgrade pip

python3 -m pip install \
    flask \
    yfinance \
    pandas \
    numpy \
    matplotlib \
    mplfinance \
    scipy \
    langchain \
    langchain-openai \
    langchain-anthropic \
    langchain-core \
    langgraph \
    openai \
    anthropic \
    ipython \
    Pillow \
    requests

# Optional: langchain-qwq (Qwen support) — may fail on Python 3.10, so swallow errors
python3 -m pip install langchain-qwq || echo "  (skipping langchain-qwq — Qwen support disabled)"

# Testing dependencies
python3 -m pip install pytest pytest-cov pytest-mock httpx

# Scheduler for background scanning
python3 -m pip install apscheduler

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python3 -c "import flask; print(f'Flask: {flask.__version__}')"
python3 -c "import yfinance; print(f'yfinance: {yfinance.__version__}')"
python3 -c "import pandas; print(f'pandas: {pandas.__version__}')"
python3 -c "import numpy; print(f'numpy: {numpy.__version__}')"
python3 -c "import pytest; print(f'pytest: {pytest.__version__}')"
python3 -c "import langchain_anthropic; print('langchain-anthropic: OK')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run web UI:         python3 web_interface.py"
echo "To run tests:          python3 -m pytest tests/ -v"
echo "To run scanner (once): python3 scanner.py --once"
echo "To run scanner (loop): python3 scanner.py --interval 14400"
