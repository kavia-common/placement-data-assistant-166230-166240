#!/bin/bash
cd /home/kavia/workspace/code-generation/placement-data-assistant-166230-166240/placement_records_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

