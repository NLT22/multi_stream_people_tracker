#Requires -Version 5.1
param(
    [Parameter(Position=0)] [string]$Command = "",
    [Parameter(Position=1)] [string]$Name = "",
    [Parameter(Position=2)] [string]$NewName = ""
)

$ClaudeDir   = Join-Path $env:USERPROFILE ".claude"
$AccountsDir = Join-Path $ClaudeDir "accounts"
$CredsFile   = Join-Path $ClaudeDir ".credentials.json"
# Displayed account info (email, organization, plan) lives here, NOT in .credentials.json.
# Swapping only credentials switches the quota/token but leaves the shown account stale.
$ConfigFile  = Join-Path $env:USERPROFILE ".claude.json"

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

function Read-TextNoBom([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $t = Get-Content $path -Raw
    if ($null -ne $t) { $t = $t.TrimStart([char]0xFEFF) }
    return $t
}

function Write-TextNoBom([string]$path, [string]$text) {
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($path, $text, $utf8NoBom)
}

# Locates the full "oauthAccount":{...} fragment via brace matching (handles nested
# objects like ccOnboardingFlags:{} and respects quoted strings). Returns @{Start;Length} or $null.
function Get-OAuthAccountSpan([string]$raw) {
    $key = [regex]::Match($raw, '"oauthAccount"\s*:\s*')
    if (-not $key.Success) { return $null }
    $braceStart = $key.Index + $key.Length
    if ($braceStart -ge $raw.Length -or $raw[$braceStart] -ne '{') { return $null }
    $depth = 0; $inStr = $false; $esc = $false
    for ($i = $braceStart; $i -lt $raw.Length; $i++) {
        $ch = $raw[$i]
        if ($inStr) {
            if ($esc) { $esc = $false }
            elseif ($ch -eq '\') { $esc = $true }
            elseif ($ch -eq '"') { $inStr = $false }
        } else {
            if ($ch -eq '"') { $inStr = $true }
            elseif ($ch -eq '{') { $depth++ }
            elseif ($ch -eq '}') {
                $depth--
                if ($depth -eq 0) {
                    return @{ Start = $key.Index; Length = ($i - $key.Index + 1) }
                }
            }
        }
    }
    return $null
}

# Returns the literal "oauthAccount":{...} fragment from .claude.json, or $null.
function Get-OAuthAccountBlock {
    $raw = Read-TextNoBom $ConfigFile
    if (-not $raw) { return $null }
    $span = Get-OAuthAccountSpan $raw
    if (-not $span) { return $null }
    return $raw.Substring($span.Start, $span.Length)
}

# Splices a saved "oauthAccount":{...} fragment back into .claude.json (first match only).
function Set-OAuthAccountBlock([string]$block) {
    $raw = Read-TextNoBom $ConfigFile
    if (-not $raw) {
        Write-Warning "$ConfigFile not found; cannot update displayed account info."
        return $false
    }
    $span = Get-OAuthAccountSpan $raw
    if (-not $span) {
        Write-Warning "No 'oauthAccount' block in $ConfigFile; displayed account info left unchanged."
        return $false
    }
    $new = $raw.Substring(0, $span.Start) + $block + $raw.Substring($span.Start + $span.Length)
    Write-TextNoBom $ConfigFile $new
    return $true
}

function Get-AccountEmail([string]$accountFile) {
    $raw = Read-TextNoBom $accountFile
    if (-not $raw) { return $null }
    $m = [regex]::Match($raw, '"emailAddress"\s*:\s*"([^"]*)"')
    if ($m.Success) { return $m.Groups[1].Value } else { return $null }
}

function Save-Profile([string]$n) {
    Validate-Name $n
    if (-not (Test-Path $CredsFile)) {
        Write-Error "No active credentials found at $CredsFile"
        exit 1
    }
    if (-not (Test-Path $AccountsDir)) { New-Item -ItemType Directory -Force $AccountsDir | Out-Null }
    Copy-Item $CredsFile (Join-Path $AccountsDir "$n.json") -Force
    $block = Get-OAuthAccountBlock
    if ($block) {
        Write-TextNoBom (Join-Path $AccountsDir "$n.account.json") $block
    } else {
        Write-Warning "Could not read 'oauthAccount' from $ConfigFile; email/organization will not switch for profile '$n'."
    }
    Set-Content (Join-Path $AccountsDir "_active") $n -Encoding utf8
    Write-Host "Saved current credentials and account info as profile '$n'."
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

    # Also swap the displayed account info (email/organization/plan) in .claude.json.
    $accountFile = Join-Path $AccountsDir "$n.account.json"
    if (Test-Path $accountFile) {
        $curBlock = Get-OAuthAccountBlock
        if ($curBlock) { Write-TextNoBom (Join-Path $AccountsDir "_backup_last.account.json") $curBlock }
        $block = (Read-TextNoBom $accountFile).Trim()
        if (Set-OAuthAccountBlock $block) {
            $email = Get-AccountEmail $accountFile
            if ($email) { Write-Host "Account info updated -> $email" }
        }
    } else {
        Write-Warning "No saved account info for '$n' (email/organization in the UI may stay stale)."
        Write-Warning "Run 'login $n' once (or 'save $n' while this account is active) to capture it."
    }

    Set-Content (Join-Path $AccountsDir "_active") $n -Encoding utf8
    Write-Host "Switched to profile '$n'. Restart Claude Code if it is currently running."
}

function WhoAmI-Profile {
    $activeFile = Join-Path $AccountsDir "_active"
    if (-not (Test-Path $activeFile)) {
        Write-Host "No active profile set. Use 'use <name>' or 'login <name>' to set one."
        return
    }
    $active = (Get-Content $activeFile -Raw).Trim()
    $profilePath = Join-Path $AccountsDir "$active.json"
    if (-not (Test-Path $profilePath)) {
        Write-Host "Active profile: $active (profile file missing - credentials may have changed)."
        return
    }
    $sub = Get-SubscriptionType $profilePath
    $email = Get-AccountEmail (Join-Path $AccountsDir "$active.account.json")
    if ($email) {
        Write-Host "Active profile: $active ($sub) - $email"
    } else {
        Write-Host "Active profile: $active ($sub)"
    }
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
    "whoami" { WhoAmI-Profile }
    default {
        Write-Host "Usage: .\switch-claude-account.ps1 {save|list|use|login|rename|whoami} [args]"
        Write-Host ""
        Write-Host "  save <name>                Save current credentials as a named profile"
        Write-Host "  list                       List all saved profiles"
        Write-Host "  use <name>                 Switch to a saved profile"
        Write-Host "  login <name>               Login via browser and save as a named profile"
        Write-Host "  rename <oldname> <newname> Rename a saved profile"
        Write-Host "  whoami                     Show the currently active profile"
        Write-Host ""
        Write-Host "Note: 'use' swaps both the token (quota) and the displayed account"
        Write-Host "info (email/organization in .claude.json). Profiles saved before this"
        Write-Host "feature lack account info - run 'login <name>' once to (re)capture it."
        Write-Host "Restart Claude Code after switching for the change to show."
        exit 1
    }
}
