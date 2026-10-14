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
    let stem = Path::new(name).file_name().and_then(|s| s.to_str()).unwrap_or(name);
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
    p.to_path_buf()
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
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

    let model_name = get("model").unwrap_or_else(|| "yolo11n".into());
    let device = match get("device").as_deref() {
        Some("cpu") => tch::Device::Cpu,
        _ => tch::Device::cuda_if_available(),
    };

    match args.mode.as_str() {
        "train" => {
            let data = get("data").ok_or_else(|| anyhow::anyhow!("data= required"))?;
            let data_path = find_cfg_path(&data);
            let cfg = TrainCfg {
                epochs: geti("epochs", 100) as usize,
                batch: geti("batch", 16) as usize,
                imgsz: geti("imgsz", 640) as u32,
                lr0: getf("lr0", 0.01),
                lrf: getf("lrf", 0.01),
                momentum: getf("momentum", 0.937),
                weight_decay: getf("weight_decay", 0.0005),
                warmup_epochs: getf("warmup_epochs", 3.0),
                optimizer: get("optimizer").unwrap_or_else(|| "auto".into()),
                cos_lr: getb("cos_lr", false),
                amp: getb("amp", false),
                device,
                seed: geti("seed", 0) as u64,
                workers: geti("workers", 8) as usize,
                close_mosaic: geti("close_mosaic", 10) as usize,
                val_interval: 1,
                project: PathBuf::from(get("project").unwrap_or_else(|| "runs".into())),
                name: get("name").unwrap_or_else(|| "train".into()),
                exist_ok: getb("exist_ok", false),
                patience: geti("patience", 100) as usize,
                pretrained: get("pretrained").map(PathBuf::from),
            };
            let nc_from_data = infer_nc(&data_path)?;
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
            let metrics = validate_model(&model, &ds, geti("imgsz", 640) as u32, device, &model.names)?;
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
                    println!("exported onnx → {}", path.display());
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
        let mut m = YoloModel::build(stem, None, device, false)?;
        ul_nn::pt::load_state_dict_into(&m.vs, &sd, false)?;
        let nc = m.nc;
        m.names = extract_names(&meta).unwrap_or_else(|| (0..nc).map(|i| format!("class_{i}")).collect());
        let names = m.names.clone();
        return Ok((m, nc, names));
    }
    let m = load_checkpoint(Path::new(model_name), device)?;
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
