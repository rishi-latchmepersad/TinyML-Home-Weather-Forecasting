$ErrorActionPreference = 'Stop'

$schemeLines = powercfg /L
$schemeGuids = @()

foreach ($line in $schemeLines) {
    if ($line -match 'Power Scheme GUID:\s+([a-fA-F0-9-]{36})') {
        $schemeGuids += $matches[1]
    }
}

if (-not $schemeGuids) {
    throw 'No power schemes found.'
}

foreach ($schemeGuid in $schemeGuids) {
    powercfg /SETACVALUEINDEX $schemeGuid SUB_BUTTONS LIDACTION 0 | Out-Null
    powercfg /SETDCVALUEINDEX $schemeGuid SUB_BUTTONS LIDACTION 0 | Out-Null
}

powercfg /SETACTIVE SCHEME_CURRENT | Out-Null

Write-Output 'Disabled lid close actions for all detected power schemes.'
Write-Output ''
powercfg /Q SCHEME_CURRENT SUB_BUTTONS LIDACTION
