#!/bin/sh
set -e
API_BASE_URL="${API_BASE_URL:-/yatzy/api}"
echo "window.__API_BASE_URL__='${API_BASE_URL}';" > /usr/share/nginx/apps/play/config.js
