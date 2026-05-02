#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Running ruff format..."
ruff format .

echo "Running ruff check with fixes..."
ruff check . --fix

echo "All checks passed!"

