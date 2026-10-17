//! `yolo` CLI：ultralytics 风格的 `task mode key=value...` 语法。

use std::path::{Path, PathBuf};

use clap::Parser;

use ul_engine::{load_checkpoint, predict, save_checkpoint, train, validate_model, TrainCfg};
use ul_nn::model::YoloModel;
use ul_nn::parse::Task;

#[derive(Parser)]
#[command(name = "yolo", about = "ultralytics-rust: full YOLO family in Rust")]
struct Args {
    /// 任务：detect / classify（其余任务家族见 roadmap）
    task: String,
    /// 模式：train / predict / val / export
    mode: String,
    /// key=value 参数（ultralytics 语法）
    kvs: Vec<String>,
}

/// 模型路径解析：cwd → weights/ 目录（官方权重默认下载位置）；
/// 裸 `*.pt` 名（如 yolo26n.pt）缺文件时自动从官方资产库下载。
fn resolve_model_path(name: &str) -> PathBuf {
    let p = Path::new(name);
    if p.exists() {
        return p.to_path_buf();
    }
    let alt = Path::new("weights").join(name);
    if alt.exists() {
        return alt;
    }
    if is_bare_official_weight(name) {
        let low = name.to_ascii_lowercase();
        let dir = Path::new("weights");
        let _ = std::fs::create_dir_all(dir);
        let dst = dir.join(name);
        // yolo26 系权重在 v8.4.0 release，其余在 v8.3.0；逐 tag 回退
        let tags: [ &str; 2 ] = if low.starts_with("yolo26") { ["v8.4.0", "v8.3.0"] } else { ["v8.3.0", "v8.4.0"] };
        for tag in tags {
            let url = format!("https://github.com/ultralytics/assets/releases/download/{tag}/{name}");
            println!("downloading official weights: {url}");
            let _ = std::fs::remove_file(&dst);
            if run_fetcher(&url, &dst) && valid_pt(&dst) {
                println!("saved → {}", dst.display());
                return dst;
            }
            let _ = std::fs::remove_file(&dst);
        }
        eprintln!("warning: download failed (网络受限可手动下载权重放入 weights/)");
    }
    p.to_path_buf()
}

/// 完整性校验：ultralytics .pt 为 zip 容器，必须以 PK 魔数开头且非 404 文本。
fn valid_pt(p: &Path) -> bool {
    match std::fs::read(p) {
        Ok(data) => data.len() > 1024 && data.starts_with(b"PK"),
        Err(_) => false,
    }
}

/// 官方资产库托管的权重名（yolo/yolov 前缀 + .pt）。
fn is_bare_official_weight(name: &str) -> bool {
    let low = name.to_ascii_lowercase();
    (low.starts_with("yolo") || low.starts_with("yolov")) && low.ends_with(".pt") && !name.contains('/')
}

/// 下载器：优先系统 curl（Win10 1803+ 自带），回退 PowerShell。
fn run_fetcher(url: &str, dst: &Path) -> bool {
    let curl = std::process::Command::new("curl")
        .args(["-sL", "--max-time", "600", "-o"])
        .arg(dst)
        .arg(url)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if curl && dst.exists() {
        return true;
    }
    let ps = format!(
        "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -Uri '{}' -OutFile '{}'",
        url,
        dst.display()
    );
    std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &ps])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn parse_kvs(kvs: &[String]) -> std::collections::HashMap<String, String> {
    let mut m = std::collections::HashMap::new();
    for kv in kvs {
        if let Some((k, v)) = kv.split_once('=') {
            m.insert(k.to_string(), v.to_string());
        }
    }
    m
}

/// 名字解析：磁盘文件优先；否则从嵌入注册表导出到临时目录。
fn find_cfg_path(name: &str) -> PathBuf {
    let p = Path::new(name);
    if p.exists() {
        return p.to_path_buf();
    }
    let mut stem = Path::new(name).file_name().and_then(|s| s.to_str()).unwrap_or(name);
    if stem.ends_with(".pt") {
        stem = &stem[..stem.len() - 3];
    }
    for dir in ["", "datasets/", "v8/", "11/", "12/", "26/", "v10/", "v9/", "v6/", "v5/", "v3/"] {
        let key = if dir.is_empty() { format!("models/{stem}") } else { format!("models/{dir}{stem}") };
        for cand in [key.clone(), format!("{key}.yaml")] {
            if let Some(text) = ul_core::cfg::embedded::get(&cand) {
                let out = std::env::temp_dir().join(cand.replace('/', "_"));
                std::fs::write(&out, text).ok();
                return out;
            }
        }
    }
    // 注册表未命中：回传剥掉 .pt 的惯用名（load_model_yaml 可做 scale 推断）
    PathBuf::from(stem)
}

/// Windows/MSVC：MSVC 链接器不链接无符号引用的导入库，torch_cuda.dll 不会被加载、
/// 其静态注册器不运行 → ATen dispatcher 无 CUDA 后端，cuda_if_available() 恒为
/// false、静默回退 CPU。启动时强制加载一次（幂等）。
#[cfg(windows)]
fn ensure_torch_cuda_loaded() {
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut std::ffi::c_void;
    }
    let ok = unsafe { !LoadLibraryA(c"torch_cuda.dll".as_ptr().cast()).is_null() };
    if !ok {
        log::debug!("torch_cuda.dll 加载失败（CPU 版 libtorch 或无 GPU），按 CPU 处理");
    }
}

#[cfg(not(windows))]
fn ensure_torch_cuda_loaded() {}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    ensure_torch_cuda_loaded();
    let args = Args::parse();
    let kv = parse_kvs(&args.kvs);
    let get = |k: &str| kv.get(k).cloned();
    let getf = |k: &str, d: f64| get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let geti = |k: &str, d: i64| get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let getb = |k: &str, d: bool| get(k).map(|v| v == "true" || v == "True" || v == "1").unwrap_or(d);

    let task = match args.task.as_str() {
        "detect" => Task::Detect,
        "classify" | "cls" => Task::Classify,
        "segment" | "seg" => Task::Segment,
        "pose" => Task::Pose,
        "obb" => Task::Obb,
        t => anyhow::bail!("unknown task '{t}'"),
    };
    let _ = task;

    let model_raw = get("model").unwrap_or_else(|| "yolo11n".into());
    let model_resolved = resolve_model_path(&model_raw);
    let model_name = model_resolved.to_string_lossy().to_string();
    let device = match get("device").as_deref() {
        Some("cpu") => tch::Device::Cpu,
        _ => tch::Device::cuda_if_available(),
    };

    match args.mode.as_str() {
        "train" => {
            let data = get("data").ok_or_else(|| anyhow::anyhow!("data= required"))?;
            // classify：data 为目录布局根，不做 yaml 解析
            let data_path = if args.task == "classify" || args.task == "cls" {
                PathBuf::from(&data)
            } else {
                find_cfg_path(&data)
            };
            // 增强超参透传（default.yaml Hyperparameters 段；未指定的键取官方默认值）
            let mut augment = ul_data::augment::AugmentCfg::default();
            augment.hsv_h = getf("hsv_h", augment.hsv_h);
            augment.hsv_s = getf("hsv_s", augment.hsv_s);
            augment.hsv_v = getf("hsv_v", augment.hsv_v);
            augment.degrees = getf("degrees", augment.degrees);
            augment.translate = getf("translate", augment.translate);
            augment.scale = getf("scale", augment.scale);
            augment.shear = getf("shear", augment.shear);
            augment.perspective = getf("perspective", augment.perspective);
            augment.flipud = getf("flipud", augment.flipud);
            augment.fliplr = getf("fliplr", augment.fliplr);
            augment.bgr = getf("bgr", augment.bgr);
            augment.mosaic = getf("mosaic", augment.mosaic);
            augment.mixup = getf("mixup", augment.mixup);
            augment.copy_paste = getf("copy_paste", augment.copy_paste);
            augment.cutmix = getf("cutmix", augment.cutmix);
            augment.erasing = getf("erasing", augment.erasing);
            augment.close_mosaic = geti("close_mosaic", augment.close_mosaic as i64) as usize;
            let cfg = TrainCfg {
                epochs: geti("epochs", 100) as usize,
                batch: geti("batch", 16) as usize,
                imgsz: geti("imgsz", 640) as u32,
                lr0: getf("lr0", 0.01),
                lrf: getf("lrf", 0.01),
                momentum: getf("momentum", 0.937),
                weight_decay: getf("weight_decay", 0.0005),
                warmup_epochs: getf("warmup_epochs", 3.0),
                warmup_bias_lr: getf("warmup_bias_lr", 0.1),
                optimizer: get("optimizer").unwrap_or_else(|| "auto".into()),
                cos_lr: getb("cos_lr", false),
                amp: getb("amp", true),
                device,
                seed: geti("seed", 0) as u64,
                workers: geti("workers", 8) as usize,
                augment,
                nbs: geti("nbs", 16) as usize,
                multi_scale: getb("multi_scale", false),
                cache_ram: getb("cache", false),
                val_interval: geti("val_interval", 1) as usize,
                project: PathBuf::from(get("project").unwrap_or_else(|| "runs".into())),
                name: get("name").unwrap_or_else(|| "train".into()),
                exist_ok: getb("exist_ok", false),
                patience: geti("patience", 100) as usize,
                pretrained: get("pretrained").map(PathBuf::from),
            };
            // classify：nc = 子目录数（DOTA/COCO 风格目录布局）
            let nc_from_data = if args.task == "classify" || args.task == "cls" {
                std::fs::read_dir(&data_path)
                    .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).count() as i64)
                    .unwrap_or(80)
            } else {
                infer_nc(&data_path)?
            };
            let mut model = if model_name.ends_with(".pt") && Path::new(&model_name).exists() {
                let (sd, _) = ul_nn::pt::load_pt(Path::new(&model_name))?;
                let stem = Path::new(&model_name)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("yolo11n");
                // 用原始 stem 构建（scale 从名字推断：yolov8n → n）
                let mut m = YoloModel::build(stem, Some(nc_from_data), device, true)?;
                let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false)?;
                log::info!("load {model_name}: {}", rep.summary());
                m
            } else {
                let cfg_path = find_cfg_path(&model_name);
                YoloModel::build(&cfg_path.to_string_lossy(), Some(nc_from_data), device, true)?
            };
            let report = train(&mut model, &data_path, &cfg)?;
            println!(
                "train done: {} epochs, best_fitness={:.4} mAP50-95={:.4} → {}",
                report.epochs_done,
                report.best_fitness,
                report.final_metrics.map50_95,
                report.save_dir.display()
            );
        }
        "predict" => {
            let source = get("source").ok_or_else(|| anyhow::anyhow!("source= required"))?;
            let (model, _nc, names) = load_model_for_infer(&model_name, device)?;
            let mut sources = Vec::new();
            let sp = Path::new(&source);
            if sp.is_dir() {
                let mut v: Vec<PathBuf> = std::fs::read_dir(sp)
                    .map(|rd| {
                        rd.filter_map(|e| e.ok())
                            .map(|e| e.path())
                            .filter(|p| {
                                p.extension()
                                    .and_then(|x| x.to_str())
                                    .map(|x| ["jpg", "jpeg", "png", "webp", "bmp"].contains(&x.to_lowercase().as_str()))
                                    .unwrap_or(false)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                v.sort();
                sources = v;
            } else {
                sources.push(sp.to_path_buf());
            }
            let save = if getb("save", !get("save_dir").is_none()) {
                Some(PathBuf::from(get("save_dir").unwrap_or_else(|| "runs/predict".into())))
            } else {
                None
            };
            let outs = predict(
                &model,
                &sources,
                geti("imgsz", 640) as u32,
                device,
                getf("conf", 0.25),
                save.as_deref(),
                &names,
            )?;
            for o in outs {
                println!("{}", o.path.display());
                for d in o.dets.iter().take(10) {
                    let name = names.get(d.cls as usize).map(|s| s.as_str()).unwrap_or("?");
                    println!(
                        "  {} {:.3} [{:.0},{:.0},{:.0},{:.0}]",
                        name, d.score, d.xyxy[0], d.xyxy[1], d.xyxy[2], d.xyxy[3]
                    );
                }
            }
        }
        "val" => {
            let data = get("data").ok_or_else(|| anyhow::anyhow!("data= required"))?;
            let data_path = find_cfg_path(&data);
            let (_train, val_dirs, names, kpt_dim) = ul_data::dataset::load_dataset_yaml(&data_path)?;
            let val_dir = val_dirs.first().ok_or_else(|| anyhow::anyhow!("no val split"))?;
            let ds = ul_data::dataset::YoloDataset::from_img_dir(val_dir, names, kpt_dim)?;
            let (model, _, _) = load_model_for_infer(&model_name, device)?;
            let metrics = ul_engine::validate_task(&model, &ds, geti("imgsz", 640) as u32, device, &model.names)?;
            println!(
                "val: P={:.4} R={:.4} mAP50={:.4} mAP50-95={:.4}",
                metrics.mp, metrics.mr, metrics.map50, metrics.map50_95
            );
        }
        "export" => {
            let (model, _, _) = load_model_for_infer(&model_name, device)?;
            let fmt = get("format").unwrap_or_else(|| "safetensors".into());
            match fmt.as_str() {
                "safetensors" | "pt" => {
                    let out = get("save_dir").unwrap_or_else(|| "runs/export".into());
                    save_checkpoint(&model, Path::new(&out), &model.names)?;
                    println!("exported {} → {out}/weights.safetensors", model.yaml_file);
                }
                "onnx" => {
                    let out = get("save_dir").unwrap_or_else(|| "runs/export".into());
                    let imgsz = geti("imgsz", 640) as u32;
                    let path = ul_nn::onnx::export_onnx(&model, imgsz, Path::new(&out))?;
                    // 按模型文件名命名（嵌入 yaml 的 yaml_file 可能不带 scale 后缀）
                    let stem = Path::new(&model_name)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("model");
                    let want = Path::new(&out).join(format!("{stem}.onnx"));
                    if want != path && stem != "model" {
                        std::fs::rename(&path, &want)?;
                        println!("exported onnx → {}", want.display());
                    } else {
                        println!("exported onnx → {}", path.display());
                    }
                }
                f => anyhow::bail!("unsupported export format '{f}' (supported: safetensors, onnx)"),
            }
        }
        m => anyhow::bail!("unknown mode '{m}' (train/predict/val/export)"),
    }
    Ok(())
}

fn infer_nc(data: &Path) -> anyhow::Result<i64> {
    let (_, _, names, _) = ul_data::dataset::load_dataset_yaml(data)?;
    Ok(names.len() as i64)
}

/// 推理模型加载：.pt 官方权重 / 训练产物目录（meta.json + weights.safetensors）。
fn load_model_for_infer(model_name: &str, device: tch::Device) -> anyhow::Result<(YoloModel, i64, Vec<String>)> {
    if model_name.ends_with(".pt") && Path::new(model_name).exists() {
        let (sd, meta) = ul_nn::pt::load_pt(Path::new(model_name))?;
        let stem = Path::new(model_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("yolo11n");
        // 官方发布权重的 nc 与 yaml 默认不同（如 obb=DOTA 15），从 ckpt names 推断
        let names_ckpt = extract_names(&meta);
        let nc = names_ckpt.as_ref().map(|n| n.len() as i64);
        let mut m = YoloModel::build(stem, nc, device, false)?;
        let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false);
        match &rep {
            Ok(r) => log::info!("load {model_name}: {} mismatch={:?} unexpected={:?}", r.summary(), r.shape_mismatch, r.unexpected),
            Err(e) => log::warn!("load {model_name}: {e}"),
        }
        rep?;
        let nc = m.nc;
        m.names = names_ckpt.unwrap_or_else(|| (0..nc).map(|i| format!("class_{i}")).collect());
        let names = m.names.clone();
        return Ok((m, nc, names));
    }
    // 训练产物目录优先；否则按嵌入 yaml 构建（随机权重，用于导出/烟测）
    if let Ok(m) = load_checkpoint(Path::new(model_name), device) {
        let nc = m.nc;
        let names = m.names.clone();
        return Ok((m, nc, names));
    }
    let cfg_path = find_cfg_path(model_name);
    let m = YoloModel::build(&cfg_path.to_string_lossy(), None, device, false)?;
    let nc = m.nc;
    let names = m.names.clone();
    Ok((m, nc, names))
}

/// 从 ckpt 对象树抽 model.names。
fn extract_names(meta: &ul_nn::pt::pickle::PValue) -> Option<Vec<String>> {
    use ul_nn::pt::pickle::PValue;
    let find_names = |obj: &PValue| -> Option<Vec<String>> {
        if let PValue::Object { attrs, .. } = obj {
            for (k, v) in attrs {
                if k.as_str() == Some("names") {
                    if let PValue::Dict(pairs) = v {
                        let mut out: Vec<(i64, String)> = pairs
                            .iter()
                            .filter_map(|(a, b)| Some((a.as_i64()?, b.as_str()?.to_string())))
                            .collect();
                        out.sort();
                        return Some(out.into_iter().map(|(_, n)| n).collect());
                    }
                }
            }
        }
        None
    };
    let model = meta.get("model")?;
    if let Some(n) = find_names(model) {
        return Some(n);
    }
    if let PValue::Object { attrs, .. } = model {
        for (_, v) in attrs {
            if let Some(n) = find_names(v) {
                return Some(n);
            }
        }
    }
    None
}
