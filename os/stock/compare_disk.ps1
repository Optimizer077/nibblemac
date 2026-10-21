# Compare an image with a physical disk in 4 MB blocks and list the blocks that differ.
# Must run elevated (raw disk read). Read-only.
param(
    [Parameter(Mandatory)] [int]    $DiskNumber,
    [Parameter(Mandatory)] [string] $ImagePath,
    [Parameter(Mandatory)] [string] $LogPath
)
$ErrorActionPreference = 'Stop'
Start-Transcript -Path $LogPath -Force | Out-Null
try {
    $chunk = 4MB
    $a = New-Object byte[] $chunk; $b = New-Object byte[] $chunk
    $img = [IO.File]::OpenRead($ImagePath)
    $dsk = [IO.FileStream]::new("\\.\PhysicalDrive$DiskNumber", [IO.FileMode]::Open, [IO.FileAccess]::Read,
                                [IO.FileShare]::ReadWrite, [int]$chunk, [IO.FileOptions]::None)
    [long]$off = 0; $diff = @()
    while (($n = $img.Read($a, 0, $chunk)) -gt 0) {
        $m = $dsk.Read($b, 0, $n)
        if ($m -ne $n -or -not [Linq.Enumerable]::SequenceEqual([byte[]]$a, [byte[]]$b)) { $diff += $off }
        $off += $n
    }
    $img.Close(); $dsk.Close()
    Write-Output ("compared {0:N0} MB, differing 4MB blocks: {1}" -f ($off / 1MB), $diff.Count)
    foreach ($d in $diff) { Write-Output ("DIFF at offset {0} ({1:N0} MB)" -f $d, ($d / 1MB)) }
}
catch { Write-Output "ERROR: $_" }
finally { Stop-Transcript | Out-Null }
