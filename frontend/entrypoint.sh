#!/bin/bash

set -e  # Exit immediately if a command fails

# Inject API_BASE_URL into a JavaScript file
if [ -z "${API_BASE_URL}" ]; then
  echo "Error: API_BASE_URL is not set."
  exit 1
fi

echo "window.API_BASE_URL='${API_BASE_URL}';" > /app/js/config.js

# Start the Python HTTP server
exec python3 -m http.server 8090 --bind 0.0.0.0