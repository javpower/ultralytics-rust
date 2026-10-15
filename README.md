# ultralytics-rust

**Ultralytics YOLO 全系列的纯 Rust 复刻**：训练、验证、预测、导出全链路，零 Python 依赖
（唯一的深度学习底座是 libtorch 的 Rust 绑定 `tch`，与 PyTorch 官方同一 C++ 运行时）。

```bash
# 加载官方 yolov8n.pt（纯 Rust pickle 解析器，无需 Python/官方库）直接推理
yolo detect predict model=yolov8n.pt source=coco8/images/val save=true

# COCO 格式数据训练（SGD/AdamW、warmup+线性/余弦调度、EMA、mosaic 全套增强）
yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640

# 验证（COCO mAP 10 档 + P/R，对拍官方 ap_per_class 语义）
yolo detect val data=coco8.yaml model=yolov8n.pt

# 导出：safetensors 权重包 / 纯 Rust 生成的 ONNX（onnx.checker + onnxruntime 双验证）
yolo detect export model=yolov8n.pt format=onnx
```

## 支持范围

**模型家族**（官方 YAML 驱动，`ultralytics/cfg/models/**` 直接嵌入二进制，119 个文件）：

| 家族 | detect | seg | pose | obb | cls | 说明 |
|---|---|---|---|---|---|---|
| YOLO26 | ✅ | ✅ | ✅ | ✅ | ✅ | end2end、reg_max=1、Segment26/Pose26/OBB26 |
| YOLO11 | ✅ | ✅ | ✅ | ✅ | ✅ | C3k2 / C2PSA / DWConv cls 头 |
| YOLOv8 | ✅ | ✅ | ✅ | ✅ | ✅ | C2f、ghost/p2/p6 变体 |
| YOLOv9 | ✅ | ✅ | — | — | — | RepNCSPELAN4 / ELAN1 / AConv / ADown / SPPELAN |
| YOLOv10 | ✅ | — | — | — | — | CIB / C2fCIB / SCDown / v10Detect 双头 |
| YOLOv5/v6/v3 | ✅ | — | — | — | — | 含 v3-tiny（MaxPool/ZeroPad2d） |

**48/48 个官方 YAML 全部构建+前向通过**（`cargo test -p ul-nn --test parse_all`）。
延期（roadmap）：RT-DETR（transformer 栈）、YOLO-World/YOLOE（CLIP 文本编码）、
yolo26-depth（TorchVision 桥）、SAVPE。

## 里程碑验证（全部可复现，见 `crates/*/tests`）

| 验证项 | 结果 |
|---|---|
| 官方权重加载（yolov8n.pt 298 张量 / yolo11n.pt 418 张量） | **0 缺口**（missing/shape_mismatch/unexpected 全 0） |
| 纯 Rust pickle VM 解析 torch.save 协议 2-4 | 全量解析（含 persistent storage、OrderedDict、BINGET 语义） |
| 同输入前向 vs 官方 ultralytics 8.4.157（letterbox 320 实图，84 通道全通道） | **max_diff = 0.0007** |
| 蒸馏参照：零输入 640 前向（前 4096 值） | max_diff = 6.1e-5 |
| COCO 评测：compute_ap 对拍 numpy 真值（trapz 梯形积分语义） | 完美预测 0.995 精确一致 |
| 全系列构建+前向 | 48/48 |
| ONNX 导出（v8 全路径） | onnx.checker 通过 + onnxruntime 推理成功 (1,144,2100) |
| coco8 训练冒烟（30 epoch，SGD） | best mAP50-95 = 0.376（预训练 finetune 轨迹正常收敛） |

## 与官方的能力对齐

- **parse_model**：`[from, repeats, module, args]` 逐行同构；width/depth 缩放
  （make_divisible ceil×8）、C3k2 的 m/l/x c3k 特判、SPPF legacy 补 SiLU、
  ResNetLayer 的 `[c1,c2,s,is_first,n]` 参数序
- **模块动物园**：Conv/DWConv/Conv2/GhostConv/LightConv/ConvTranspose/Focus、
  C1/C2/C2f/C3/C3x/C3f/C3Ghost/C3k/C3k2、Bottleneck/GhostBottleneck/BottleneckCSP、
  SPP/SPPF、RepC3/RepConv/RepNCSPELAN4/ELAN1、AConv/ADown/SPPELAN、CBLinear/CBFuse、
  SCDown/CIB/C2fCIB、RepVGGDW、PSA 全家（Attention/PSABlock/PSA/C2PSA/C2fPSA）、
  A2C2f（YOLOv12 Area-Attention）、HGStem/HGBlock、ResNetLayer、MaxPool2d/ZeroPad2d
- **五任务头**：Detect（含 one2one 双头）、Segment(+Proto26)、Pose(+Pose26 的
  cv4_kpts/cv4_sigma 结构)、OBB(+OBB26 原始角度)、v10Detect、Classify、SemanticSegment
- **损失**：v8DetectionLoss（TAL + BCE + 成对 CIoU + DFL 量化 CE）、E2E 双头、
  分类 CE+label smoothing；权重 box=7.5/cls=0.5/dfl=1.5 与官方一致
- **TAL 分配器**：官方 8.4 语义——gt 主序 (bs,m,A)、topk scatter 计数、
  一锚多 gt 保留最大 IoU、小框放大 stride_val、norm align parity
- **数据管线**：mosaic4 / mixup / 随机透视（translate/scale/shear/degrees/perspective）/
  HSV（真 HSV 转换）/ fliplr（COCO17 关键点交换表）/ flipud / BGR / close_mosaic /
  letterbox（114 灰边）
- **评测**：COCO 10 档 IoU + trapz 101 点插值 + smooth-F1 最优点 P/R（ap_per_class 忠实移植）

## 权重入口

- **官方 `.pt` 直接加载**（pickle VM + zip storage 重建 + 按名对位 VarStore）
- 训练产物：`weights.safetensors` + `meta.json`（task/nc/names/stride/yaml_file），
  可用 `load_checkpoint` 复原继续训练/推理
- 导出：`format=safetensors` 或 `format=onnx`

## Crate 结构

```
crates/
├── ul-core         纯逻辑：几何/框运算、模型+数据集 YAML 解析（官方 cfg build.rs 嵌入）、
│                   COCO mAP（trapz 对拍 numpy）、纯 CPU NMS
├── ul-nn           张量层：模块动物园（参数名与 ultralytics state_dict 逐字对齐）、
│                   parse_model、官方 .pt 加载（pickle VM）、TAL、ONNX 导出
├── ul-data         图像 IO、YOLO 数据集（detect/pose）、增强管线
├── ul-engine       v8/E2E 损失、trainer（auto 优化器/EMA/调度/裁剪）、validator、predictor
└── ultralytics-rs  `yolo` CLI（train/predict/val/export，key=value 语法）
```

## 构建与运行

```bash
# 需 libtorch 2.11（与 tch 0.24 硬配对）：设 LIBTORCH 指向本机 libtorch
export LIBTORCH=/path/to/libtorch
cargo build --release

# 官方权重（首次自动下载到 weights/）
./target/release/yolo detect predict model=yolov8n.pt source=<图片或目录> save=true
```

CUDA：libtorch CUDA 版即可（`tch` 运行时探测），无卡自动回退 CPU。

## 已知边界（诚实清单）

- 训练损失量级与官方有差异（绝对值大 ~10×，收敛方向一致）——DFL 权重化与
  分支归一化的细节仍在对齐；`best` 检查点按 fitness 正确保存
- `mosaic=0` 等 T超参 CLI 透传尚未接全（AugmentCfg 默认值生效）
- ONNX 导出输出为**原始头**（1, reg_max*4+nc, A；yolov8n@320 = (1,144,2100)），
  官方默认导出含 DFL 积分=(1,84,A)——解码差异，图内 DFL 积分在 roadmap
- v9 渐进式（CBLinear/CBFuse 已有模块，yolov9 yaml 组装待验证）、
  RT-DETR/YOLO-World/YOLOE/depth 未实现
- AMP（fp16 autocast）、多卡、rect 训练未实现
- pickle VM 支持 torch.save 协议 2-4 的 torch 标准子集；
  `.pt` 若含非常规 opcode 会显式报错（不支持静默错位）

## 许可

AGPL-3.0（与 Ultralytics 上游一致——本项目为 Ultralytics 的衍生实现）。
官方 cfg YAML 与模型设计版权归 Ultralytics。