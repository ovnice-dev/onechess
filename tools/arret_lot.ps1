# Arrete la file des SPRT en lot et tous ses pilotes (les tests GitHub
# continuent ; relancer `bash ci/file_lot.sh 4` les reprend).
# Attention : ne pas filtrer sur « eval », les options des tests en contiennent.
Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
    Where-Object { $_.CommandLine -match 'ci/(sprt_lot|file_lot)\.sh' -and $_.CommandLine -notmatch 'bash\.exe" -c ' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false; "arrete : $($_.ProcessId)" }
