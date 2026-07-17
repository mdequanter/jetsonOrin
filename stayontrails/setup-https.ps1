# Set up a locally-trusted HTTPS certificate for freewalkFlask using mkcert.
# Run this on the Windows machine that RUNS the server.
#
# After running it, freewalkFlask.py picks up .\freewalk-cert.pem + freewalk-key.pem
# automatically. Install the printed rootCA.pem on your phone once, and the page
# loads over HTTPS with no browser warning, so the camera + GPS work.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cert = Join-Path $here "freewalk-cert.pem"
$key  = Join-Path $here "freewalk-key.pem"

if (-not (Get-Command mkcert -ErrorAction SilentlyContinue)) {
    Write-Host "mkcert is not installed. Install it, then re-run this script:" -ForegroundColor Yellow
    Write-Host "  choco install mkcert      (Chocolatey)"
    Write-Host "  scoop install mkcert      (Scoop)"
    exit 1
}

# Detect the LAN IPv4 address (the interface with a default gateway).
$lanip = (Get-NetIPConfiguration |
    Where-Object { $null -ne $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq "Up" } |
    Select-Object -First 1 -ExpandProperty IPv4Address).IPAddress
if (-not $lanip) { $lanip = "127.0.0.1" }

# Short hostname (e.g. jetson-desktop) plus its mDNS .local alias.
$hostShort = $env:COMPUTERNAME.ToLower()
$hostLocal = "$hostShort.local"

Write-Host "==> Installing the mkcert local CA..."
mkcert -install

Write-Host "==> Generating certificate for: localhost 127.0.0.1 $lanip $hostShort $hostLocal"
mkcert -cert-file $cert -key-file $key localhost 127.0.0.1 $lanip $hostShort $hostLocal

$caroot = (mkcert -CAROOT).Trim()
Write-Host ""
Write-Host "Done. freewalkFlask.py will use these automatically:"
Write-Host "  cert: $cert"
Write-Host "  key : $key"
Write-Host ""
Write-Host "Trust the CA on your phone (one time):"
Write-Host "  Root CA file:  $caroot\rootCA.pem"
Write-Host ""
Write-Host "  Android: copy rootCA.pem to the phone ->"
Write-Host "           Settings -> Security -> Encryption & credentials ->"
Write-Host "           Install a certificate -> CA certificate -> pick rootCA.pem"
Write-Host "  iOS:     send rootCA.pem to the phone -> install the profile ->"
Write-Host "           Settings -> General -> About -> Certificate Trust Settings ->"
Write-Host "           enable full trust for the mkcert CA"
Write-Host ""
Write-Host "Then start the server and open on the phone:"
Write-Host "  python freewalkFlask.py"
Write-Host "  https://${hostLocal}:5003   (preferred - survives IP changes)"
Write-Host "  https://${lanip}:5003       (fallback if mDNS/.local doesn't resolve)"
