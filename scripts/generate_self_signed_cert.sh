#!/usr/bin/env bash
# Generates a self-signed TLS cert/key for the reverse proxy (see
# docker-compose.override.yml and nginx/nginx.conf). Run this on whichever
# host will actually serve traffic, using that host's public IP/hostname -
# the certificate's SAN must match what the browser connects to, or it will
# refuse to even offer the "proceed anyway" option.
#
# Usage: scripts/generate_self_signed_cert.sh <ip-or-hostname>
set -euo pipefail

HOST="${1:?Usage: $0 <ip-or-hostname>}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/certs"
mkdir -p "$OUT_DIR"

# Detect whether $HOST is an IP (needs an IP: SAN) or a name (needs a DNS: SAN) -
# modern browsers ignore the certificate CN and only trust the SAN entry.
if [[ "$HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  SAN="IP:$HOST"
else
  SAN="DNS:$HOST"
fi

# The doubled leading "//" (rather than a bare "/") stops Git-Bash-on-Windows
# from rewriting -subj's value into a Windows path (e.g.
# "C:/Program Files/Git/CN=..."); OpenSSL's subject parser tolerates the
# resulting empty leading component. A no-op on Linux/macOS.
openssl req -x509 -nodes -newkey rsa:2048 -days 825 \
  -keyout "$OUT_DIR/server.key" \
  -out "$OUT_DIR/server.crt" \
  -subj "//CN=$HOST" \
  -addext "subjectAltName=$SAN"

echo "Wrote $OUT_DIR/server.crt and $OUT_DIR/server.key for $HOST ($SAN)"
echo "Restart the reverse-proxy service to pick them up: docker compose restart reverse-proxy"
