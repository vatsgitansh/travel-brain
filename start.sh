#!/usr/bin/env bash
python -m uvicorn travel_brain.api.app:app --host 0.0.0.0 --port ${PORT:-10000}
