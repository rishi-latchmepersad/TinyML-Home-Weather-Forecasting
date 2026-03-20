param(
    [Parameter(Mandatory = $true)]
    [string]$LinuxCommand,
    [string]$Distro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"

wsl --shutdown
Start-Sleep -Seconds 2
wsl -d $Distro -- bash -lc $LinuxCommand
