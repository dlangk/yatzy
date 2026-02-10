#!/bin/sh
set -e

if [ -z "${API_BASE_URL}" ]; then
  echo "Error: API_BASE_URL is not set."
  exit 1
fi

echo "window.__API_BASE_URL__='${API_BASE_URL}';" > /usr/share/nginx/html/config.js

exec nginx -g 'daemon off;'
