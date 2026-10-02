param(
    [Parameter(Mandatory = $true)]
    [string]$RtspUrl,

    [Parameter(Mandatory = $true)]
    [ValidateSet("forward_fall", "backward_fall", "left_fall", "right_fall", "upper_body_occluded", "lower_body_occluded", "walking", "sitting", "standing")]
    [string]$Pose,

    [ValidateSet("fall", "notfall")]
    [string]$Category = "fall",

    [int]$DurationSeconds = 30,

    [string]$OutputRoot = "data/fall_test"
)

$ErrorActionPreference = "Stop"
if ($DurationSeconds -le 0) {
    throw "DurationSeconds must be greater than zero."
}

$today = Get-Date -Format "yyyyMMdd"
$recordedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDirectory = Join-Path $OutputRoot "$today\$Category"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$outputPath = Join-Path $outputDirectory "${recordedAt}_${Pose}.mp4"

Write-Host "Recording: $RtspUrl"
Write-Host "Output:    $outputPath"
Write-Host "Duration:  ${DurationSeconds}s"

& ffmpeg -hide_banner -loglevel warning -y `
    -rtsp_transport tcp -i $RtspUrl -t $DurationSeconds `
    -map 0:v:0 -vf "fps=30" -r 30 -c:v libx264 -preset veryfast `
    -pix_fmt yuv420p -movflags +faststart $outputPath

if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg exited with code $LASTEXITCODE"
}

Write-Host "Saved: $outputPath"
