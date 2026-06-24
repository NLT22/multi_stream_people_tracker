#Requires -Version 5.1
param(
    [Parameter(Position=0)] [string]$Command = "",
    [Parameter(Position=1)] [string]$Name = "",
    [Parameter(Position=2)] [string]$NewName = ""
)

$ClaudeDir   = Join-Path $env:USERPROFILE ".claude"
$AccountsDir = Join-Path $ClaudeDir "accounts"
$CredsFile   = Join-Path $ClaudeDir ".credentials.json"

function Validate-Name([string]$n) {
    if ($n -match '[\s/\\]') {
        Write-Error "Profile name must be a single word (no spaces or slashes)."
        exit 1
    }
}

function Get-SubscriptionType([string]$FilePath) {
    try {
        $j = Get-Content $FilePath -Raw | ConvertFrom-Json
        $sub = $j.claudeAiOauth.subscriptionType
        if ($sub) { return $sub } else { return "unknown" }
    } catch { return "unknown" }
}

function Save-Profile([string]$n) {
    Validate-Name $n
    if (-not (Test-Path $CredsFile)) {
        Write-Error "No active credentials found at $CredsFile"
        exit 1
    }
    if (-not (Test-Path $AccountsDir)) { New-Item -ItemType Directory -Force $AccountsDir | Out-Null }
    Copy-Item $CredsFile (Join-Path $AccountsDir "$n.json") -Force
    Set-Content (Join-Path $AccountsDir "_active") $n -Encoding utf8
    Write-Host "Saved current credentials as profile '$n'."
}

function List-Profiles {
    if (-not (Test-Path $AccountsDir)) { New-Item -ItemType Directory -Force $AccountsDir | Out-Null }
    $activeFile = Join-Path $AccountsDir "_active"
    $active = if (Test-Path $activeFile) { (Get-Content $activeFile -Raw).Trim() } else { "" }

    $profiles = Get-ChildItem "$AccountsDir\*.json" -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -ne "_backup_last.json" }

    Write-Host "Saved profiles:"
    if (-not $profiles) {
        Write-Host "  No saved profiles. Use 'save <name>' or 'login <name>' to create one."
        return
    }
    foreach ($p in $profiles) {
        $pname = [System.IO.Path]::GetFileNameWithoutExtension($p.Name)
        $sub   = Get-SubscriptionType $p.FullName
        $marker = if ($pname -eq $active) { "*" } else { " " }
        Write-Host ("  {0} {1,-20} ({2})" -f $marker, $pname, $sub)
    }
}

function Use-Profile([string]$n) {
    Validate-Name $n
    $profile = Join-Path $AccountsDir "$n.json"
    if (-not (Test-Path $profile)) {
        Write-Error "Profile '$n' not found. Run 'list' to see available profiles."
        exit 1
    }
    if (-not (Test-Path $AccountsDir)) { New-Item -ItemType Directory -Force $AccountsDir | Out-Null }
    if (Test-Path $CredsFile) {
        Copy-Item $CredsFile (Join-Path $AccountsDir "_backup_last.json") -Force
    }
    Copy-Item $profile $CredsFile -Force
    Set-Content (Join-Path $AccountsDir "_active") $n -Encoding utf8
    Write-Host "Switched to profile '$n'. Restart Claude Code if it is currently running."
}

function Rename-Profile([string]$oldname, [string]$newname) {
    Validate-Name $oldname
    Validate-Name $newname
    $src = Join-Path $AccountsDir "$oldname.json"
    $dst = Join-Path $AccountsDir "$newname.json"
    if (-not (Test-Path $src)) {
        Write-Error "Profile '$oldname' not found. Run 'list' to see available profiles."
        exit 1
    }
    if (Test-Path $dst) {
        Write-Error "Profile '$newname' already exists. Choose a different name."
        exit 1
    }
    Copy-Item $src $dst -Force
    Remove-Item $src -Force
    $activeFile = Join-Path $AccountsDir "_active"
    if ((Test-Path $activeFile) -and ((Get-Content $activeFile -Raw).Trim() -eq $oldname)) {
        Set-Content $activeFile $newname -Encoding utf8
    }
    Write-Host "Renamed profile '$oldname' to '$newname'."
}

function Login-Profile([string]$n) {
    Validate-Name $n
    Write-Host "Opening browser login for profile '$n'..."
    try {
        & claude auth login
    } catch [System.Management.Automation.CommandNotFoundException] {
        Write-Error "Login failed: 'claude' not found on PATH. Is Claude Code installed?"
        exit 1
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Login failed (exit code $LASTEXITCODE). Credentials unchanged."
        exit 1
    }
    Save-Profile $n
    Write-Host "Profile '$n' saved. Now active."
}

# --- dispatch ---
switch ($Command.ToLower()) {
    "save"  { if (-not $Name) { Write-Error "Usage: .\switch-claude-account.ps1 save <name>"; exit 1 }; Save-Profile $Name }
    "list"  { List-Profiles }
    "use"   { if (-not $Name) { Write-Error "Usage: .\switch-claude-account.ps1 use <name>"; exit 1 }; Use-Profile $Name }
    "login"  { if (-not $Name) { Write-Error "Usage: .\switch-claude-account.ps1 login <name>"; exit 1 }; Login-Profile $Name }
    "rename" {
        if (-not $Name -or -not $NewName) { Write-Error "Usage: .\switch-claude-account.ps1 rename <oldname> <newname>"; exit 1 }
        Rename-Profile $Name $NewName
    }
    default {
        Write-Host "Usage: .\switch-claude-account.ps1 {save|list|use|login|rename} [args]"
        Write-Host ""
        Write-Host "  save <name>                Save current credentials as a named profile"
        Write-Host "  list                       List all saved profiles"
        Write-Host "  use <name>                 Switch to a saved profile"
        Write-Host "  login <name>               Login via browser and save as a named profile"
        Write-Host "  rename <oldname> <newname> Rename a saved profile"
        exit 1
    }
}
