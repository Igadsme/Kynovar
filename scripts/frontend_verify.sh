#!/bin/sh
set -eu

if [ "${KYNOVAR_FRONTEND_DOCKER:-0}" = "1" ]; then
  check_dir=$(mktemp -d "${TMPDIR:-/tmp}/kynovar-frontend.XXXXXX")
  trap 'rm -rf "$check_dir"' EXIT HUP INT TERM
  tar -C frontend --exclude=node_modules --exclude=.next -cf - . | tar -C "$check_dir" -xf -
  docker run --rm -v "$check_dir:/app" -w /app node:22-alpine sh -c \
    'npm ci && npm run typecheck && npm run build'
else
  npm --prefix frontend ci
  npm --prefix frontend run typecheck
  npm --prefix frontend run build
fi
