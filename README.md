# ultralytics-rust

**Ultralytics YOLO 全系列的纯 Rust 复刻**：训练、验证、预测、导出全链路，零 Python 依赖
（唯一的深度学习底座是 libtorch 的 Rust 绑定 `tch`，与 PyTorch 官方同一 C++ 运行时）。

```bash
# 加载官方 yolov8n.pt（纯 Rust pickle 解析器，无需 Python/官方库）直接推理
yolo detect predict model=yolov8n.pt source=coco8/images/val save=true

# COCO 格式数据训练（SGD/AdamW、warmup+线性/余弦调度、EMA、mosaic 全套增强）
yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640
# 增强超参全套透传（未指定的键取官方 default.yaml 默认值）
yolo detect train data=coco8.yaml model=yolov8n.pt mosaic=0.0 mixup=0.2 fliplr=0.3 hsv_v=0.5

# 验证（COCO mAP 10 档 + P/R，对拍官方 ap_per_class 语义）
yolo detect val data=coco8.yaml model=yolov8n.pt

# 导出：safetensors 权重包 / 纯 Rust 生成的 ONNX（图内解码对齐官方默认导出）
yolo detect export model=yolov8n.pt format=onnx imgsz=640
```

## 支持范围

**模型家族**（官方 YAML 驱动，`ultralytics/cfg/models/**` 直接嵌入二进制，119 个文件）：

| 家族 | detect | seg | pose | obb | cls | 说明 |
|---|---|---|---|---|---|---|
| YOLO26 | ✅ | ✅ | ✅ | ✅ | ✅ | end2end、reg_max=1、Segment26/Pose26/OBB26 |
| YOLO11 | ✅ | ✅ | ✅ | ✅ | ✅ | C3k2 / C2PSA / DWConv cls 头 |
| YOLOv8 | ✅ | ✅ | ✅ | ✅ | ✅ | C2f、ghost/p2/p6 变体 |
| YOLOv9 | ✅ | — | — | — | — | GELAN 全家（t/s/c/m/e）含 CBLinear/CBFuse 前向 |
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
| **v8DetectionLoss 三分量 vs 官方 8.4.157**（同输入同权重，`loss_parity.rs`） | **box 6.6e-5 / cls 2.2e-4 / dfl 4.6e-5 相对差**；TAL 分配 fg=44 一致、target_scores_sum 7.874 vs 7.877 |
| **任务损失对拍**（`loss_parity_task.rs`，官方 v8Seg/Pose/OBB Loss 同输入同权重） | **全部分量对齐**：obb box 3.4418 逐位 / cls 7.8725 vs 7.8723 / dfl / angle 0.4115 逐位；pose 11.1475 精确 / kobj 1.2383 精确；seg 0.9715 = 官方 0.9715（×7.5=7.286）逐位；detect 分量三任务全达标。对拍累计揪出并修复 8 处真实 bug（旋转角点半宽、probiou t2、obb 角度缩放、seg nm/npr 缩放反置、seg 面积 x2y2、seg 除法广播、pose 全局归一、kobj logits BCE） |
| COCO 评测：compute_ap 对拍 numpy 真值（trapz 梯形积分语义） | 完美预测 0.995 精确一致 |
| 全系列构建+前向 | 48/48 |
| **ONNX 导出（v5-v11 全路径，图内 DFL+dist2bbox+sigmoid）** | onnx.checker 通过；onnxruntime 对拍官方 eval 前向：yolov8n 320→(1,84,2100) **max_diff 6.4e-4**、640→(1,84,8400) **9.5e-4**；yolo11n（含 C2PSA 注意力）**7.8e-4** |
| **ONNX e2e 导出（v10/yolo26，图内两级 topk → (1,300,6)）** | checker + onnxruntime 通过；分数降序、值域校验 ✓（官方 v10 权重资产缺失，数值对拍以 y8/y11 为准） |
| **四任务训练闭环**（detect/segment/pose/obb） | 合成标签冒烟全部收敛运行：seg 走 BCE/crop/area 掩码损失，pose 走 OKS 式 KeypointLoss+kobj BCE，obb 走旋转 TAL+probiou+rbox2dist+角度损失 |
| **评测覆盖** | detect=box AP；seg=mask IoU AP；pose=OKS AP；obb=probiou AP（官方 match_predictions/ap_per_class 语义） |
| coco8 训练冒烟（30 epoch，自动 AdamW，默认增强） | best mAP50-95 = **0.381**（修复前 0.376）；关增强直训 0.399 |

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
- **损失**：v8DetectionLoss（TAL + BCE + 成对 CIoU + DFL 量化 CE）、E2E 双头
  （o2m topk10 + o2o topk1，对齐 E2EDetectLoss）、reg_max=1 的归一化 L1 路径
  （官方 use_dfl=False 分支）、分类 CE+label smoothing；
  权重 box=7.5/cls=0.5/dfl=1.5 与官方一致；CIoU 为官方语义
  （rho²=(dx²+dy²)/4、c²=对角线平方 cw²+ch²）；BCE 用 softplus 稳定形式
- **TAL 分配器**：官方 8.4 语义——gt 主序 (bs,m,A)、topk scatter 计数、
  一锚多 gt 保留最大 IoU、小框放大 stride_val=stride[1]、norm align parity、
  CIoU clamp≥0 后取 align（负 CIoU 在 ^β 下会翻正）
- **数据管线**：mosaic4 / mixup / 随机透视（translate/scale/shear/degrees/perspective）/
  HSV（真 HSV 转换）/ fliplr（COCO17 关键点交换表）/ flipud / BGR / close_mosaic /
  letterbox（114 灰边）；以上 14 项超参全部经 CLI 透传
- **评测**：COCO 10 档 IoU + trapz 101 点插值 + smooth-F1 最优点 P/R（ap_per_class 忠实移植）
- **训练器**：官方三参数组（bias 无 wd+warmup_bias_lr=0.1 起步 / norm 无 wd /
  weight 有 wd×batch÷nbs）手写 SGD+AdamW（含 AdamW 解耦 wd、bias correction）、
  nbs 梯度累积（每 max(1,⌈nbs/batch⌉) 步 clip+step）、multi-scale（[0.5,1.5]×imgsz 32 对齐）、
  auto 优化器（total_iters≤10000 → AdamW+lr_fit，否则 SGD）、
  warmup（lr 0→lr0、momentum 0.8→目标）、线性/余弦调度、裁剪 grad=10、
  EMA 逐迭代 ramp（0.9999·(1−e^(−i/2000))）、close_mosaic 末段关闭、
  fitness 最优检查点 + patience 早停 + metrics.jsonl（box/cls/dfl 等分量）
- **任务损失**：v8SegmentationLoss（proto 双线性上采样 + einsum 掩码 +
  bbox 裁剪 BCE/面积 ÷fg 数，×box gain）、v8PoseLoss（格单位 kpt 解码 +
  KeypointLoss((2σ)²·area·2) + kobj BCE，×pose=12/kobj=1）、v8OBBLoss
  （RotatedTaskAlignedAssigner 角点投影 + probiou(floor=0.01) + rbox2dist DFL +
  sin²(2Δθ)·宽高比权重角度损失，×angle=1）
- **任务评测**：seg=掩码 IoU 矩阵 AP（proto@coef 上采样 + bbox 裁剪）、
  pose=OKS 矩阵 AP（COCO17 σ）、obb=probiou 矩阵 AP；均走官方
  match_predictions（10 档 IoU）+ ap_per_class

## 权重入口

- **官方 `.pt` 直接加载**（pickle VM + zip storage 重建 + 按名对位 VarStore）
- 训练产物：`weights.safetensors` + `meta.json`（task/nc/names/stride/yaml_file），
  可用 `load_checkpoint` 复原继续训练/推理
- 导出：`format=safetensors` 或 `format=onnx`（解码后输出，语义同官方默认导出）

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

- **训练侧未实现**：AMP（fp16 autocast——tch 绑定未暴露 autocast 上下文）、
  多卡 DDP（tch 未暴露 c10d）；**rect 推理已实现**（predict 32 对齐矩形
  letterbox），rect 训练（宽高比分批）未实现
- **增强**：copy_paste（flip 模式，IoA<0.3 过滤 + 多边形掩码贴像素）、cutmix、
  erasing（classify 管线）已实现并透传（CLI copy_paste=/cutmix=/erasing=）；
  copy_paste 的 mixup 模式未实现
- **任务面**：detect/classify/segment/pose/obb 训练+评测闭环全通；**三任务损失
  全部分量与官方 8.4.157 逐位/精确对齐**（seg/pose/obb 的 detect 分量 + seg/kobj/
  angle/pose 附加分量）；官方 pt 推理加载支持 ckpt nc 覆盖（obb=DOTA 15 等）；
  **classify 训练支持目录布局数据集**（root/<class>/<img>，类按目录名排序）；
  yolo26 训练语义未与官方曲线对拍
- **ONNX 导出**：v5-v11 标准 Detect（图内解码）+ v10/yolo26 e2e（图内两级
  topk→(1,300,6)）+ PSA 注意力族 + **seg/pose/obb 头全图内解码**
  （seg (1,116,A)、pose (1,56,A) 含 kpt 图内 (y·2+anchor−0.5)·stride 解码+v=σ(v)、
  obb (1,20,A) 含 dist2rbox 旋转解码+(sigmoid−0.25)π 角度；均对拍官方
  ≤1e-3，checker+ORT 通过）；A2C2f（v12）显式报错（roadmap）
- **数据集缓存**：labels 解析结果持久化 cache.json（构造时全量解析一次，
  二次启动零解析；条目数不匹配自动失效重建）
- **classify 训练**：支持目录布局数据集（data=<root>，子目录即类）；COCO 风格
  txt 标签不适用于 classify（官方同样分目录）
- **曲线级对拍**：与官方 trainer 的逐 epoch 收敛对比脚本见
  `scripts/curve_parity.py`（同参数同数据双方训练），尚未跑 COCO 全量基准
- **pickle VM**：支持 torch.save 协议 2-4 的 torch 标准子集；
  `.pt` 若含非常规 opcode 会显式报错（不支持静默错位）

## 许可

AGPL-3.0（与 Ultralytics 上游一致——本项目为 Ultralytics 的衍生实现）。
官方 cfg YAML 与模型设计版权归 Ultralytics。
