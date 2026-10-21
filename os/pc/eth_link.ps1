# Direct PC <-> Pi Ethernet cable (no router): PC 10.42.0.1/24, Pi 10.42.0.2/24 (set on the Pi).
# No gateway on this link, so the PC keeps using Wi-Fi for internet. Must run elevated.
#   eth_link.ps1 -LogPath <file>          configure
#   eth_link.ps1 -LogPath <file> -Undo    back to DHCP (when the cable goes to a router again)
param(
    [Parameter(Mandatory)] [string] $LogPath,
    [string] $Alias = 'Ethernet',
    [switch] $Undo
)
$ErrorActionPreference = 'Stop'
Start-Transcript -Path $LogPath -Force | Out-Null
try {
    if ($Undo) {
        Get-NetIPAddress -InterfaceAlias $Alias -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object IPAddress -eq '10.42.0.1' | Remove-NetIPAddress -Confirm:$false
        Set-NetIPInterface -InterfaceAlias $Alias -AddressFamily IPv4 -Dhcp Enabled
        Write-Output "Undo done: $Alias back to DHCP."
    } else {
        Set-NetIPInterface -InterfaceAlias $Alias -AddressFamily IPv4 -Dhcp Disabled
        Get-NetIPAddress -InterfaceAlias $Alias -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Remove-NetIPAddress -Confirm:$false
        New-NetIPAddress -InterfaceAlias $Alias -IPAddress 10.42.0.1 -PrefixLength 24 | Out-Null
        Write-Output "Configured: $Alias = 10.42.0.1/24, no gateway."
    }
    Get-NetIPAddress -InterfaceAlias $Alias -AddressFamily IPv4 | Format-Table IPAddress, PrefixLength, PrefixOrigin -AutoSize | Out-String | Write-Output
}
catch { Write-Output "ERROR: $_" }
finally { Stop-Transcript | Out-Null }
