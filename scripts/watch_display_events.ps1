$ErrorActionPreference = 'SilentlyContinue'

$logDir = Join-Path $PSScriptRoot '..\diagnostics'
$null = New-Item -ItemType Directory -Path $logDir -Force

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $logDir "display_flash_watch_$timestamp.log"
$pidPath = Join-Path $logDir 'display_flash_watch.pid'

$systemProviders = @(
    'Display',
    'Desktop Window Manager',
    'Microsoft-Windows-Kernel-Power',
    'Microsoft-Windows-Kernel-PnP',
    'Microsoft-Windows-UserPnp',
    'Microsoft-Windows-WHEA-Logger',
    'igccservice',
    'igfx',
    'igfxCUIService2.0.0.0',
    'Monitor',
    'nhi'
)

$applicationProviders = @(
    '.NET Runtime',
    'Application Error',
    'Desktop Window Manager',
    'igccservice',
    'igfxCUIService2.0.0.0',
    'Intel.GraphicsSoftware.Service',
    'Windows Error Reporting'
)

$seen = New-Object 'System.Collections.Generic.HashSet[string]'

function Write-Log {
    param([string]$Text)

    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'
    Add-Content -Path $logPath -Value "[$stamp] $Text"
}

Write-Log "watcher started; pid=$PID"
Set-Content -Path $pidPath -Value $PID

while ($true) {
    try {
        $since = (Get-Date).AddSeconds(-15)

        $systemEvents = Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $since } -ErrorAction SilentlyContinue
        foreach ($event in $systemEvents) {
            if ($event.ProviderName -notin $systemProviders -and $event.Id -notin 41, 42, 105, 107, 187, 219, 4101, 6008) {
                continue
            }

            $key = "System:$($event.RecordId)"
            if ($seen.Add($key)) {
                $message = ($event.Message -replace '\s+', ' ').Trim()
                Write-Log "System Id=$($event.Id) Provider=$($event.ProviderName) Level=$($event.LevelDisplayName) Time=$($event.TimeCreated.ToString('o')) Message=$message"
            }
        }

        $applicationEvents = Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $since } -ErrorAction SilentlyContinue
        foreach ($event in $applicationEvents) {
            if ($event.ProviderName -notin $applicationProviders) {
                continue
            }

            $key = "Application:$($event.RecordId)"
            if ($seen.Add($key)) {
                $message = ($event.Message -replace '\s+', ' ').Trim()
                Write-Log "Application Id=$($event.Id) Provider=$($event.ProviderName) Level=$($event.LevelDisplayName) Time=$($event.TimeCreated.ToString('o')) Message=$message"
            }
        }
    }
    catch {
        Write-Log "watcher loop error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds 2
}
