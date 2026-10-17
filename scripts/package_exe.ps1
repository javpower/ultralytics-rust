# 打包 Windows 绿色分发版（体积极小，运行时按需自举）：
#   yolo.exe      启动器（检测显卡 → 首次运行自动下载 CUDA/CPU 运行时）
#   yolo_core.exe 推理/训练引擎
# 用法：先 cargo build --release，再运行本脚本；产物 dist\yolo-win64-green\ 打 zip 即可分发
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Rel  = "target\release"
$Dist = Join-Path $Root "dist\yolo-win64-green"

foreach ($f in @("yolo.exe", "yolo_core.exe")) {
    $src = Join-Path $Root (Join-Path $Rel $f)
    if (-not (Test-Path $src)) { throw "未找到 $src，请先 cargo build --release" }
}
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
Copy-Item (Join-Path $Root (Join-Path $Rel "yolo.exe"))      (Join-Path $Dist "yolo.exe") -Force
Copy-Item (Join-Path $Root (Join-Path $Rel "yolo_core.exe")) (Join-Path $Dist "yolo_core.exe") -Force

@'
YOLO 绿色版（Windows x64，本体 <10MB）

首次运行任意命令时自动安装运行时：
  - 有 NVIDIA 显卡：自动下载 CUDA 版（约 2.8GB，需一次联网）
  - 无显卡：自动下载 CPU 版（约 300MB）

训练:  yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
推理:  yolo detect predict model=yolo26n.pt source=image.jpg save=true
验证:  yolo detect val   data=coco8.yaml model=yolo26n.pt

- 裸权重名（如 yolo26n.pt）缺文件时自动从官方资产库下载
- 开发者模式：设置 LIBTORCH 环境变量后跳过运行时自举
'@ | Out-File (Join-Path $Dist "README.txt") -Encoding utf8

$size = [math]::Round((Get-ChildItem $Dist -Recurse | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "打包完成：$Dist（$size MB）— 打 zip 即可分发"
