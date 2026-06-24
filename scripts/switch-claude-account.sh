#!/usr/bin/env bash
set -euo pipefail

CLAUDE_DIR="$HOME/.claude"
ACCOUNTS_DIR="$CLAUDE_DIR/accounts"
CREDS_FILE="$CLAUDE_DIR/.credentials.json"

# JSON helper: extracts .claudeAiOauth.subscriptionType from a credentials file
get_subscription_type() {
  local file="$1"
  if command -v jq &>/dev/null; then
    jq -r '.claudeAiOauth.subscriptionType // "unknown"' "$file" 2>/dev/null || echo "unknown"
  else
    python3 -c "
import json, sys
try:
    d = json.load(open('$file'))
    print(d.get('claudeAiOauth', {}).get('subscriptionType', 'unknown'))
except Exception:
    print('unknown')
"
  fi
}

validate_name() {
  local name="$1"
  if [[ "$name" =~ [[:space:]/\\] || "$name" == *"'"* ]]; then
    echo "Error: Profile name must be a single word (no spaces, slashes, or quotes)." >&2
    exit 1
  fi
}

save_profile() {
  local name="$1"
  validate_name "$name"
  if [[ ! -f "$CREDS_FILE" ]]; then
    echo "Error: No active credentials found at $CREDS_FILE" >&2
    exit 1
  fi
  mkdir -p "$ACCOUNTS_DIR"
  cp "$CREDS_FILE" "$ACCOUNTS_DIR/$name.json"
  echo "$name" > "$ACCOUNTS_DIR/_active"
  echo "Saved current credentials as profile '$name'."
}

list_profiles() {
  mkdir -p "$ACCOUNTS_DIR"
  local active=""
  [[ -f "$ACCOUNTS_DIR/_active" ]] && active=$(cat "$ACCOUNTS_DIR/_active")

  local found=0
  echo "Saved profiles:"
  for f in "$ACCOUNTS_DIR"/*.json; do
    [[ "$f" == *"_backup_last.json" ]] && continue
    [[ ! -f "$f" ]] && continue
    found=1
    local pname
    pname=$(basename "$f" .json)
    local sub
    sub=$(get_subscription_type "$f")
    if [[ "$pname" == "$active" ]]; then
      printf "  * %-20s (%s)\n" "$pname" "$sub"
    else
      printf "    %-20s (%s)\n" "$pname" "$sub"
    fi
  done

  if [[ $found -eq 0 ]]; then
    echo "  No saved profiles. Use 'save <name>' or 'login <name>' to create one."
  fi
}

use_profile() {
  local name="$1"
  validate_name "$name"
  local profile="$ACCOUNTS_DIR/$name.json"
  if [[ ! -f "$profile" ]]; then
    echo "Error: Profile '$name' not found. Run 'list' to see available profiles." >&2
    exit 1
  fi
  mkdir -p "$ACCOUNTS_DIR"
  [[ -f "$CREDS_FILE" ]] && cp "$CREDS_FILE" "$ACCOUNTS_DIR/_backup_last.json"
  cp "$profile" "$CREDS_FILE"
  echo "$name" > "$ACCOUNTS_DIR/_active"
  echo "Switched to profile '$name'. Restart Claude Code if it is currently running."
}

rename_profile() {
  local oldname="$1"
  local newname="$2"
  validate_name "$oldname"
  validate_name "$newname"
  local src="$ACCOUNTS_DIR/$oldname.json"
  local dst="$ACCOUNTS_DIR/$newname.json"
  if [[ ! -f "$src" ]]; then
    echo "Error: Profile '$oldname' not found. Run 'list' to see available profiles." >&2
    exit 1
  fi
  if [[ -f "$dst" ]]; then
    echo "Error: Profile '$newname' already exists. Choose a different name." >&2
    exit 1
  fi
  cp "$src" "$dst"
  rm "$src"
  local active=""
  [[ -f "$ACCOUNTS_DIR/_active" ]] && active=$(cat "$ACCOUNTS_DIR/_active")
  if [[ "$active" == "$oldname" ]]; then
    echo "$newname" > "$ACCOUNTS_DIR/_active"
  fi
  echo "Renamed profile '$oldname' to '$newname'."
}

login_profile() {
  local name="$1"
  validate_name "$name"
  if ! command -v claude &>/dev/null; then
    echo "Error: 'claude' not found on PATH. Is Claude Code installed?" >&2
    exit 1
  fi
  echo "Opening browser login for profile '$name'..."
  if ! claude auth login; then
    echo "Error: Login failed. Credentials unchanged." >&2
    exit 1
  fi
  save_profile "$name"
  echo "Profile '$name' saved. Now active."
}

# --- dispatch ---
CMD="${1:-}"
NAME="${2:-}"

case "$CMD" in
  save)
    [[ -z "$NAME" ]] && { echo "Usage: $0 save <name>" >&2; exit 1; }
    save_profile "$NAME"
    ;;
  list)
    list_profiles
    ;;
  use)
    [[ -z "$NAME" ]] && { echo "Usage: $0 use <name>" >&2; exit 1; }
    use_profile "$NAME"
    ;;
  login)
    [[ -z "$NAME" ]] && { echo "Usage: $0 login <name>" >&2; exit 1; }
    login_profile "$NAME"
    ;;
  rename)
    NEWNAME="${3:-}"
    [[ -z "$NAME" || -z "$NEWNAME" ]] && { echo "Usage: $0 rename <oldname> <newname>" >&2; exit 1; }
    rename_profile "$NAME" "$NEWNAME"
    ;;
  *)
    echo "Usage: $0 {save|list|use|login|rename} [args]"
    echo ""
    echo "  save <name>              Save current credentials as a named profile"
    echo "  list                     List all saved profiles"
    echo "  use <name>               Switch to a saved profile"
    echo "  login <name>             Login via browser and save as a named profile"
    echo "  rename <oldname> <newname>  Rename a saved profile"
    exit 1
    ;;
esac
