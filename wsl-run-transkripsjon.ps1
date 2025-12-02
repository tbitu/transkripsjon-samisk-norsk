
# Filename: run-transkripsjon.ps1
# Usage: Run from an elevated PowerShell window on Windows 10/11 Home or Pro

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$containerName = "transkripsjon"
$imageName = "ghcr.io/tbitu/transkripsjon-samisk-norsk:latest"
$restartRequired = $false

function Require-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Warning "This script must run from an elevated PowerShell session."
        Write-Host "Right-click PowerShell and choose 'Run as administrator', then run the script again."
        exit 1
    }
}

function Check-Virtualization {
    Write-Host "=== Checking hardware virtualization support ==="
    try {
        $cpuInfo = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
        $virtEnabled = $cpuInfo.VirtualizationFirmwareEnabled
    } catch {
        $virtEnabled = $null
    }

    if ($virtEnabled -ne $true) {
        Write-Warning "Virtualization is disabled in firmware. Enable Intel VT-x/AMD-V (often called SVM) in BIOS/UEFI, then rerun this script."
        Write-Host "Windows Home relies on this setting for WSL2."
        exit 1
    }

    Write-Host "Firmware virtualization is enabled."
}

function Check-NvidiaDriver {
    Write-Host "=== Checking NVIDIA driver ==="
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Warning "nvidia-smi was not found. Install the latest NVIDIA Game Ready or Studio driver with WSL support, then rerun this script."
        Write-Host "Download: https://www.nvidia.com/Download/index.aspx"
        exit 1
    }

    $driverInfo = & $nvidiaSmi.Path | Select-String "Driver Version" | Select-Object -First 1
    if ($driverInfo) {
        Write-Host $driverInfo.ToString().Trim()
    } else {
        Write-Host "NVIDIA driver detected."
    }
}

function Ensure-WindowsFeature {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FeatureName,

        [Parameter(Mandatory = $true)]
        [string] $FriendlyName
    )

    $feature = Get-WindowsOptionalFeature -Online -FeatureName $FeatureName
    if ($feature.State -ne "Enabled") {
        Write-Host "Enabling $FriendlyName..."
        Enable-WindowsOptionalFeature -Online -FeatureName $FeatureName -All -NoRestart | Out-Null
        $script:restartRequired = $true
    } else {
        Write-Host "$FriendlyName already enabled."
    }
}

function Ensure-WSLFeatures {
    Write-Host "=== Ensuring required Windows features are enabled ==="
    Ensure-WindowsFeature -FeatureName "Microsoft-Windows-Subsystem-Linux" -FriendlyName "Windows Subsystem for Linux"
    Ensure-WindowsFeature -FeatureName "VirtualMachinePlatform" -FriendlyName "Virtual Machine Platform"

    if ($script:restartRequired) {
        Write-Warning "Windows features were just enabled. Restart your PC, then rerun this script to continue."
        exit 0
    }
}

function Ensure-WSL2Default {
    Write-Host "=== Setting WSL2 as the default version ==="
    try {
        wsl --set-default-version 2 | Out-Null
    } catch {
        Write-Warning "Unable to set WSL2 as default. Run 'wsl --set-default-version 2' manually and rerun the script."
        throw
    }
}

function Ensure-UbuntuDistro {
    Write-Host "=== Checking Ubuntu WSL distribution ==="
    $wslList = wsl --list --quiet 2>$null
    $hasUbuntu = $wslList | Where-Object { $_ -match '^Ubuntu' }

    if (-not $hasUbuntu) {
        Write-Host "Ubuntu was not found. Installing Ubuntu from the Microsoft Store..."
        wsl --install -d Ubuntu
        Write-Warning "When installation finishes, Windows will ask for a restart. After reboot, open the 'Ubuntu' app once to create a username/password, then rerun this script."
        exit 0
    }

    Write-Host "Ubuntu detected. Setting it as the default WSL distribution."
    wsl --set-default Ubuntu | Out-Null

    Write-Host "Verifying that Ubuntu has completed its first-run setup..."
    $testCommand = "echo wsl-ready"
    $null = & wsl -d Ubuntu -- bash -lc $testCommand
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Ubuntu has not finished its initial setup. Launch the 'Ubuntu' app manually, finish the on-screen steps, then rerun this script."
        exit 1
    }
}

function Check-UbuntuGpuAccess {
    Write-Host "=== Verifying that WSL can see the NVIDIA GPU ==="
    try {
        $null = & wsl -d Ubuntu -- nvidia-smi
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi returned exit code $LASTEXITCODE"
        }
        Write-Host "GPU visible inside WSL."
    } catch {
        Write-Warning "nvidia-smi failed inside WSL. Ensure the NVIDIA driver with WSL support is installed and reboot, then rerun the script."
        throw
    }
}

function Ensure-DockerDesktop {
    Write-Host "=== Checking Docker Desktop installation ==="
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Warning "Docker CLI not found. Install Docker Desktop for Windows (WSL2 backend) from https://www.docker.com/products/docker-desktop, enable WSL integration for Ubuntu, then rerun the script."
        exit 1
    }

    try {
        $serverVersion = docker version --format '{{.Server.Version}}' 2>$null
        if (-not $serverVersion) {
            throw "Docker daemon not reachable"
        }
        Write-Host "Docker Desktop server version $serverVersion"
    } catch {
        Write-Warning "Docker CLI is installed but the daemon is not reachable. Start Docker Desktop, wait until it says 'Docker Desktop is running', then rerun the script."
        exit 1
    }

    $wslStatus = wsl --list --verbose | Select-String "docker-desktop" | Select-Object -First 1
    if (-not $wslStatus -or ($wslStatus.ToString() -notmatch 'Running')) {
        Write-Warning "Docker Desktop WSL integration is not running. In Docker Desktop settings, enable 'Use the WSL 2 based engine' and ensure Ubuntu is checked under 'Resources > WSL Integration'."
        Write-Host "After applying the changes, rerun this script."
        exit 1
    }
}

function Pull-Image {
    Write-Host "=== Pulling Docker image $imageName ==="
    docker pull $imageName
}

function Ensure-CleanContainer {
    Write-Host "=== Preparing container '$containerName' ==="
    $existing = docker ps -a --filter "name=^/$containerName$" --format '{{.ID}}'
    if ($existing) {
        Write-Host "Removing existing container $containerName..."
        docker rm -f $containerName | Out-Null
    }
}

function Start-Container {
    Write-Host "=== Starting container with GPU support ==="
    docker run --gpus all --restart unless-stopped -d -p 5000:5000 --name $containerName $imageName | Out-Null
    Write-Host "Container started. Waiting for services..."
    Start-Sleep -Seconds 8
}

function Validate-ContainerGpu {
    Write-Host "=== Validating GPU access inside container ==="
    $null = docker exec $containerName nvidia-smi
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed inside the container ($LASTEXITCODE)."
    }
    Write-Host "GPU check succeeded."
}

function Launch-WebApp {
    Write-Host "=== Opening http://localhost:5000 in your browser ==="
    Start-Process "http://localhost:5000"
    Write-Host "Transkripsjon service is starting up. The first request downloads models and may take a few minutes."
}

try {
    Require-Admin
    Check-Virtualization
    Check-NvidiaDriver
    Ensure-WSLFeatures
    Ensure-WSL2Default
    Ensure-UbuntuDistro
    Check-UbuntuGpuAccess
    Ensure-DockerDesktop
    Pull-Image
    Ensure-CleanContainer
    Start-Container
    Validate-ContainerGpu
    Launch-WebApp
    Write-Host "=== All steps completed successfully ==="
    Write-Host "To stop the service, run: docker rm -f $containerName"
} catch {
    Write-Error $_
    Write-Warning "The script exited early because of the error shown above. Fix the issue and run the script again."
    exit 1
}
