#!/bin/bash

# Inject API_BASE_URL into a JavaScript file
echo "window.API_BASE_URL='${API_BASE_URL}';" > /app/js/config.js

# Start the Python HTTP server
exec python3 -m http.server 8090