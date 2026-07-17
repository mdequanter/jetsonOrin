#!/usr/bin/env bash
#
# Set up a locally-trusted HTTPS certificate for freewalkFlask using mkcert.
# Run this on the machine that RUNS the server (e.g. the Jetson Orin).
#
# After running it, freewalkFlask.py picks up ./freewalk-cert.pem + freewalk-key.pem
# automatically. Install the printed rootCA.pem on your phone once, and the page
# loads over HTTPS with no browser warning, so the camera + GPS work.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT="$HERE/freewalk-cert.pem"
KEY="$HERE/freewalk-key.pem"

if ! command -v mkcert >/dev/null 2>&1; then
  cat <<'EOF'
mkcert is not installed. Install it, then re-run this script:

  # Debian / Ubuntu / Jetson (arm64):
  sudo apt-get install -y libnss3-tools
  curl -JLO "https://dl.filippo.io/mkcert/latest?for=linux/arm64"
  chmod +x mkcert-v*-linux-arm64
  sudo mv mkcert-v*-linux-arm64 /usr/local/bin/mkcert

  # On a regular Linux PC use  ...?for=linux/amd64  instead.
EOF
  exit 1
fi

# Detect the LAN IPv4 address this host is reachable at.
LAN_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
LAN_IP="${LAN_IP:-127.0.0.1}"

# Detect the short hostname (e.g. jetson-desktop) and add its mDNS .local alias.
HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
HOST_LOCAL="${HOST_SHORT}.local"

echo "==> Installing the mkcert local CA (may prompt for sudo)..."
mkcert -install

echo "==> Generating certificate for: localhost 127.0.0.1 ${LAN_IP} ${HOST_SHORT} ${HOST_LOCAL}"
mkcert -cert-file "$CERT" -key-file "$KEY" localhost 127.0.0.1 "$LAN_IP" "$HOST_SHORT" "$HOST_LOCAL"

CAROOT="$(mkcert -CAROOT)"
cat <<EOF

Done. freewalkFlask.py will use these automatically:
  cert: $CERT
  key : $KEY

Trust the CA on your phone (one time):
  Root CA file:  $CAROOT/rootCA.pem

  Android: copy rootCA.pem to the phone ->
           Settings -> Security -> Encryption & credentials ->
           Install a certificate -> CA certificate -> pick rootCA.pem
  iOS:     AirDrop/email rootCA.pem to the phone -> install the profile ->
           Settings -> General -> About -> Certificate Trust Settings ->
           enable full trust for the mkcert CA

Then start the server and open on the phone:
  python3 freewalkFlask.py
  https://${HOST_LOCAL}:5003   (preferred — survives IP changes)
  https://${LAN_IP}:5003       (fallback if mDNS/.local doesn't resolve)
EOF
