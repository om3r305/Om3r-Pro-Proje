#!/usr/bin/env bash
# NIULAI4 pilot on a free EU Linux VM (Ubuntu 22.04/24.04, x86_64 or arm64). Run as root:
#   sudo bash setup.sh <git-commit>
# Installs chrony + a pinned Deno, checks out the repo at <git-commit>, creates an unprivileged
# service user and a systemd unit. It does NOT copy the ingest key or the approval: the operator
# places those by hand (see README.md), and the service refuses to start without them.
set -euo pipefail

COMMIT="${1:?usage: setup.sh <git-commit>}"
DENO_VERSION="2.1.4"
REPO="https://github.com/om3r305/Om3r-Pro-Proje.git"
APP=/opt/niulai4-pilot
OUT=/var/lib/niulai4-pilot
ETC=/etc/niulai4-pilot
USER_NAME=niulai4

[ "$(id -u)" -eq 0 ] || { echo "run as root"; exit 1; }

apt-get update -y
apt-get install -y --no-install-recommends chrony git unzip curl ca-certificates
systemctl enable --now chrony

case "$(uname -m)" in
  x86_64) TRIPLE=x86_64-unknown-linux-gnu ;;
  aarch64) TRIPLE=aarch64-unknown-linux-gnu ;;
  *) echo "unsupported arch $(uname -m)"; exit 1 ;;
esac
TMP=$(mktemp -d)
BASE="https://github.com/denoland/deno/releases/download/v${DENO_VERSION}"
curl -fsSL -o "$TMP/deno.zip" "$BASE/deno-${TRIPLE}.zip"
curl -fsSL -o "$TMP/deno.zip.sha256sum" "$BASE/deno-${TRIPLE}.zip.sha256sum"
( cd "$TMP" && echo "$(awk '{print $1}' deno.zip.sha256sum)  deno.zip" | sha256sum -c - )
unzip -o -q "$TMP/deno.zip" -d /usr/local/bin && chmod 755 /usr/local/bin/deno
rm -rf "$TMP"
deno --version | head -1

id "$USER_NAME" >/dev/null 2>&1 || useradd --system --home-dir "$OUT" --shell /usr/sbin/nologin "$USER_NAME"
install -d -o "$USER_NAME" -g "$USER_NAME" -m 750 "$OUT"
install -d -o root -g "$USER_NAME" -m 750 "$ETC"

if [ ! -d "$APP/.git" ]; then git clone --no-checkout --filter=blob:none "$REPO" "$APP"; fi
git -C "$APP" fetch --quiet origin "$COMMIT"
git -C "$APP" -c advice.detachedHead=false checkout --quiet --force "$COMMIT"
chown -R root:root "$APP"
echo "$COMMIT" > "$ETC/COMMIT"

install -m 644 "$APP/research/niulai4_pilot/deploy/niulai4-pilot.service" /etc/systemd/system/niulai4-pilot.service
systemctl daemon-reload

# Cache dependencies as the service user so the hardened unit never downloads code at start.
sudo -u "$USER_NAME" env DENO_DIR="$OUT/.deno" deno cache "$APP/research/niulai4_pilot/main.ts"

echo
echo "Selftest (records nothing; refuses until the approval file is in $OUT):"
sudo -u "$USER_NAME" env DENO_DIR="$OUT/.deno" deno run --allow-net --allow-read --allow-write --allow-run=chronyc,df \
  "$APP/research/niulai4_pilot/main.ts" --selftest --out "$OUT" || true
echo
echo "Next: place $ETC/ingest.key and $OUT/APPROVED_RULES_FINGERPRINT (README.md), stop the PC pilot,"
echo "then: systemctl enable --now niulai4-pilot"
