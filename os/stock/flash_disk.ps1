# Write a raw .img to a whole physical disk, then read it back and verify SHA-256.
# Must run elevated. Refuses to run unless the disk matches the expected name, size and bus,
# so a renumbered disk can never be overwritten by mistake.
param(
    [Parameter(Mandatory)] [int]    $DiskNumber,
    [Parameter(Mandatory)] [string] $ExpectedName,
    [Parameter(Mandatory)] [long]   $ExpectedSize,
    [Parameter(Mandatory)] [string] $ImagePath,
    [Parameter(Mandatory)] [string] $ExpectedSha256,
    [Parameter(Mandatory)] [string] $LogPath,
    [switch] $VerifyOnly
)
$ErrorActionPreference = 'Stop'
Start-Transcript -Path $LogPath -Force | Out-Null
$exitCode = 0
$tookOffline = $false
try {
    $disk = Get-Disk -Number $DiskNumber
    if ($disk.FriendlyName -ne $ExpectedName -or $disk.Size -ne $ExpectedSize -or $disk.BusType -ne 'USB' -or $disk.IsBoot -or $disk.IsSystem) {
        throw "Disk $DiskNumber is '$($disk.FriendlyName)' size $($disk.Size) bus $($disk.BusType) - does not match, aborting."
    }
    $imgLen = (Get-Item $ImagePath).Length
    if ($imgLen % 512 -ne 0) { throw "Image size $imgLen is not a multiple of 512." }
    Write-Output "Target OK: disk $DiskNumber $($disk.FriendlyName). Image $imgLen bytes."

    Set-Disk -Number $DiskNumber -IsOffline $true   # dismounts volumes so raw writes are allowed
    $tookOffline = $true
    $chunk = 4MB
    $buf = New-Object byte[] $chunk
    if (-not $VerifyOnly) {
    $src = [IO.File]::OpenRead($ImagePath)
    $dst = [IO.FileStream]::new("\\.\PhysicalDrive$DiskNumber", [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite,
                                [IO.FileShare]::ReadWrite, [int]$chunk, [IO.FileOptions]::WriteThrough)
    $done = 0; $t0 = Get-Date
    while (($n = $src.Read($buf, 0, $chunk)) -gt 0) {
        $dst.Write($buf, 0, $n); $done += $n
        if ($done % 512MB -eq 0) { Write-Output ("written {0:N0} MB" -f ($done / 1MB)) }
    }
    $dst.Flush(); $dst.Close(); $src.Close()
    Write-Output ("Write done: {0:N0} MB in {1:N0} s" -f ($done / 1MB), ((Get-Date) - $t0).TotalSeconds)
    }

    # Read back exactly the image length and hash it.
    $sha = [Security.Cryptography.SHA256]::Create()
    $rd = [IO.FileStream]::new("\\.\PhysicalDrive$DiskNumber", [IO.FileMode]::Open, [IO.FileAccess]::Read,
                               [IO.FileShare]::ReadWrite, [int]$chunk, [IO.FileOptions]::None)
    [long]$left = $imgLen
    while ($left -gt 0) {
        $want = [int][Math]::Min([long]$chunk, $left)
        $n = $rd.Read($buf, 0, $want)
        if ($n -le 0) { throw "Short read-back at $($imgLen - $left)" }
        $null = $sha.TransformBlock($buf, 0, $n, $null, 0); $left -= $n
    }
    $null = $sha.TransformFinalBlock($buf, 0, 0); $rd.Close()
    $got = ([BitConverter]::ToString($sha.Hash) -replace '-', '').ToLower()
    Write-Output "Read-back SHA-256: $got"
    if ($got -ne $ExpectedSha256.ToLower()) { throw "VERIFY FAILED: expected $ExpectedSha256" }
    Write-Output "VERIFY OK"
}
catch {
    Write-Output "ERROR: $_"
    $exitCode = 1
}
finally {
    # Release the raw handles first (an exception can leave them open), then bring the disk
    # back online only if this script took it offline: after a failed identity check
    # $DiskNumber may name another disk, which must be left exactly as it was.
    foreach ($s in @($src, $dst, $rd)) { if ($s) { $s.Dispose() } }
    if ($tookOffline) {
        try { Set-Disk -Number $DiskNumber -IsOffline $false } catch { Write-Output "could not bring disk online: $_" }
    }
    Stop-Transcript | Out-Null
}
exit $exitCode   # non-zero on any failure (VERIFY FAILED included), for callers that check it
