# Claude Account Switcher

Switch between multiple Claude Code accounts without losing chat history.
Chat history is shared across all profiles — only the active credentials change.

## Setup

**Linux/macOS:**
```bash
chmod +x scripts/switch-claude-account.sh
```

**Windows:** No setup needed (run with `.\scripts\switch-claude-account.ps1`).

## Commands

| Command | Description |
|---------|-------------|
| `save <name>` | Save current logged-in account as a named profile |
| `list` | Show all saved profiles; `*` marks the active one |
| `use <name>` | Switch to a saved profile |
| `login <name>` | Open browser login, then save as a named profile |

## Quick Start

```bash
# Save your current account (e.g. personal)
bash scripts/switch-claude-account.sh save personal

# Log in to a second account and save it
bash scripts/switch-claude-account.sh login work

# Switch back to personal
bash scripts/switch-claude-account.sh use personal

# See all profiles
bash scripts/switch-claude-account.sh list
```

## Profile Storage

Profiles are stored in `~/.claude/accounts/`. Each is a snapshot of
`~/.claude/.credentials.json`. A backup of the previous credentials is
always saved to `_backup_last.json` before any switch.

## Notes

- Restart Claude Code after switching profiles for the change to take effect.
- Credentials are stored with the same security as the existing `.credentials.json`.
