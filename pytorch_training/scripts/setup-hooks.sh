#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Installing pre-commit..."
pip install pre-commit

echo "Installing git hooks..."
pre-commit install

echo "Done! Pre-commit hooks are now active."
echo "Run 'pre-commit run --all-files' to check all files manually."

