# Arret propre de tout ce qui tourne pour Onechess, sans rien perdre.
#
#  1. L'etiquetage recoit sa demande d'arret (data\STOP_LABEL). Il termine
#     les lignes deja distribuees et note dans <mois>\REPRISE la ligne exacte
#     ou reprendre.
#  2. Les matchs (SPRT, A/B) sont arretes. Leurs fichiers d'etat gardent
#     toutes les paires terminees ; seule la paire en cours sera rejouee.
#     Les entrainements et l'autojeu sont arretes aussi.
#  3. La conversion PGN est arretee. Le mois en cours sera refait : il n'est
#     ecrit qu'a la fin, sous un nom provisoire.
#  4. Verification : plus aucun python, moteur ni Stockfish.
#
#   powershell -ExecutionPolicy Bypass -File tools\arret_propre.ps1

$ErrorActionPreference = 'Continue'
Set-Location C:\Onechess

function Procs($motif) {
    Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
        Where-Object { $_.CommandLine -match $motif }
}

# 1. Etiquetage : arret propre, reprise exacte
if (Procs 'label\.py') {
    New-Item -ItemType File -Force data\STOP_LABEL | Out-Null
    Write-Output "demande d'arret envoyee a l'etiquetage..."
    $t0 = Get-Date
    while ((Procs 'label\.py') -and ((Get-Date) - $t0).TotalSeconds -lt 300) { Start-Sleep -Seconds 5 }
    if (Procs 'label\.py') { Write-Output "ATTENTION : l'etiquetage ne s'est pas arrete en 5 min" }
    else { Write-Output ("etiquetage arrete proprement en {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds) }
}
Get-ChildItem data\nnue_elite -Recurse -Filter REPRISE -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Output "  reprise prevue : $($_.Directory.Name), ligne $(Get-Content $_.FullName)" }

# 2. Matchs, entrainements, autojeu
foreach ($m in @('arena\.py', 'train\.py', 'datagen\.py')) {
    $p = @(Procs $m)
    if ($p.Count) {
        $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Write-Output "$($p.Count) processus $m arretes"
    }
}

# 3. Conversion
$p = @(Procs 'pgn_to_csv')
if ($p.Count) {
    $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Output "conversion arretee"
}

# La demande d'arret ne doit pas survivre : elle arreterait net le prochain
# etiquetage.
Remove-Item data\STOP_LABEL -ErrorAction SilentlyContinue

# 3 bis. Moteurs adverses du tournoi d'ancrage : arret_propre ne visait que
#        les notres. Un Leorik avait survecu au tournoi du 13-09.
$adverses = @('Leorik*','weiss*','avalanche*','reckless*','RubiChess*')
foreach ($a in $adverses) {
    $p = @(Get-Process -Name $a -ErrorAction SilentlyContinue)
    if ($p.Count) {
        $p | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Output "$($p.Count) processus $a arretes"
    }
}

# 4. Verification (les processus enfants meurent avec leur job Windows)
Start-Sleep -Seconds 5
$reste = @(Get-Process | Where-Object { $_.ProcessName -match 'python|onechess|stockfish' })
if ($reste.Count) {
    Write-Output "RESTE ENCORE :"
    $reste | Group-Object ProcessName | ForEach-Object { Write-Output "  $($_.Name) : $($_.Count)" }
}
else { Write-Output "plus rien ne tourne : 0 python, 0 moteur, 0 Stockfish" }
