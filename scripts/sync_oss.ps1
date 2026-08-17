# sync_oss.ps1 — Sync everest-an/M1 (private dev) -> AwareLiquid/M1 (public mirror)
#
# Design:
#   - everest-an/M1 is the private development repo (full history, some secrets in history).
#   - AwareLiquid/M1 is the public open-source mirror (single clean history).
#   - This script copies the CURRENT tree (never history) from dev to mirror,
#     drops excluded paths, applies the org rename, commits once, pushes.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\sync_oss.ps1
#   (requires `gh` CLI authenticated as an account with access to both repos)
#
# Excluded from the mirror (sensitive / not-open-source):
#   scripts/remote_*.py        GPU-box SSH password
#   session_alice.capsule      runtime session data
#   data/                      training data binaries (reproducible via prepare_data.py)
#   checkpoints/               model weights (published separately via GitHub Releases)
#   sync script itself         references the private repo name

param(
    [string]$WorkDir = "$env:TEMP\m1_oss_sync",
    [string]$CommitMsg = "sync: mirror latest dev tree"
)

$ErrorActionPreference = "Stop"

$exclude = @(
    "scripts/remote_exec.py",
    "scripts/remote_download.py",
    "scripts/remote_download2.py",
    "scripts/remote_upload_snap.py",
    "session_alice.capsule",
    "data",
    "checkpoints",
    "scripts/sync_oss.ps1"
)

function Sync-Tree {
    param([string]$src, [string]$dst)
    foreach ($item in Get-ChildItem -Path $src -Force | Where-Object { $_.Name -ne ".git" }) {
        $rel = $item.FullName.Substring($src.Length).TrimStart("\", "/").Replace("\", "/")
        $excluded = $false
        foreach ($ex in $exclude) {
            if ($rel -eq $ex -or $rel.StartsWith("$ex/") -or $rel.StartsWith("$ex\")) {
                $excluded = $true; break
            }
        }
        if ($excluded) { Write-Host "  EXCLUDED $rel"; continue }
        Copy-Item -Path $item.FullName -Destination $dst -Recurse -Force
    }
}

# 1. Update dev source
if (-not (Test-Path "$WorkDir\dev")) {
    gh repo clone everest-an/M1 "$WorkDir\dev" -- --quiet
} else {
    git -C "$WorkDir\dev" pull -q --ff-only origin main
}
Write-Host "dev at $(git -C "$WorkDir\dev" log --oneline -1)"

# 2. Update mirror
if (-not (Test-Path "$WorkDir\mirror")) {
    gh repo clone AwareLiquid/M1 "$WorkDir\mirror" -- --quiet
} else {
    git -C "$WorkDir\mirror" pull -q --ff-only origin main
}

# 3. Wipe mirror working tree (keep .git) and re-copy from dev
Get-ChildItem -Path "$WorkDir\mirror" -Force | Where-Object { $_.Name -ne ".git" } |
    Remove-Item -Recurse -Force
Sync-Tree -src "$WorkDir\dev" -dst "$WorkDir\mirror"

# 4. Apply org rename: everest-an -> AwareLiquid in all text files
$renamed = 0
$textFiles = Get-ChildItem -Path "$WorkDir\mirror" -Recurse -File | Where-Object {
    $_.Extension -match "^\.(md|py|sh|html|txt|json|yml|yaml|tex|cfg|ini|css|js|ts)$"
}
foreach ($f in $textFiles) {
    $c = [System.IO.File]::ReadAllText($f.FullName)
    if ($c.Contains("everest-an")) {
        $c = $c.Replace("github.com/everest-an/M1", "github.com/AwareLiquid/M1")
        $c = $c.Replace("everest-an/M1", "AwareLiquid/M1")
        [System.IO.File]::WriteAllText($f.FullName, $c)
        $renamed++
    }
}
Write-Host "org rename applied to $renamed files"

# 5. Commit + push
git -C "$WorkDir\mirror" add -A
$status = git -C "$WorkDir\mirror" status --porcelain
if ($status) {
    git -C "$WorkDir\mirror" -c user.name="EverestAn" -c user.email="everest9812@gmail.com" commit -q -m $CommitMsg
    git -C "$WorkDir\mirror" push -q origin main
    Write-Host "pushed: $(git -C "$WorkDir\mirror" log --oneline -1)"
} else {
    Write-Host "no changes to sync"
}
