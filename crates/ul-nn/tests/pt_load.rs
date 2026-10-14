//! 官方 .pt 权重加载验证（需要 weights/ 目录有官方权重）。

use std::path::Path;
use std::sync::OnceLock;

static WEIGHTS: OnceLock<()> = OnceLock::new();

fn ensure_weights() {
    WEIGHTS.get_or_init(|| {
        std::fs::create_dir_all("weights").ok();
        for f in ["yolo11n.pt", "yolov8n.pt"] {
            let p = Path::new("weights").join(f);
            if !p.exists() {
                let url = format!("https://github.com/ultralytics/assets/releases/download/v8.3.0/{f}");
                println!("downloading {f}...");
                let st = std::process::Command::new("curl")
                    .args(["-sL", "-o", p.to_str().unwrap(), &url])
                    .status()
                    .expect("curl");
                assert!(st.success(), "download {f} failed");
            }
        }
    });
}

#[test]
fn load_official_checkpoints_zero_gap() {
    ensure_weights();
    for (file, model) in [("yolo11n.pt", "yolo11n"), ("yolov8n.pt", "yolov8n")] {
        let (sd, _meta) = ul_nn::pt::load_pt(Path::new(&format!("weights/{file}"))).unwrap();
        let m = ul_nn::YoloModel::build(model, None, tch::Device::Cpu, false).unwrap();
        let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
        println!("{file}: sd={} {}", sd.len(), rep.summary());
        assert!(
            rep.shape_mismatch.is_empty(),
            "{file} shape mismatches: {:?}",
            &rep.shape_mismatch[..3.min(rep.shape_mismatch.len())]
        );
        assert!(rep.missing.is_empty(), "{file} missing: {:?}", &rep.missing[..3.min(rep.missing.len())]);
        assert!(
            rep.unexpected.is_empty(),
            "{file} unexpected: {:?}",
            &rep.unexpected[..3.min(rep.unexpected.len())]
        );
    }
}

/// Rust 前向 vs Python ultralytics 前向：相同零输入下的原始输出对拍。
#[test]
fn forward_parity_vs_python() {
    ensure_weights();
    // Python 参考输出（py312 生成，见 scripts/parity_ref.py）
    let ref_path = Path::new("E:/ultralytics-rust/weights/yolo11n_ref.json");
    if !ref_path.exists() {
        println!("SKIP: no python reference {ref_path:?} (run scripts/parity_ref.py)");
        return;
    }
    let js: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ref_path).unwrap()).unwrap();
    let (sd, _) = ul_nn::pt::load_pt(Path::new("E:/ultralytics-rust/weights/yolo11n.pt")).unwrap();
    let m = ul_nn::YoloModel::build("yolo11n", None, tch::Device::Cpu, false).unwrap();
    ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    let x = tch::Tensor::zeros([1, 3, 640, 640], (tch::Kind::Float, tch::Device::Cpu));
    let out = match m.forward(&x, false).unwrap() {
        ul_nn::model::ModelOutput::Detect(ul_nn::head::HeadOutput::Inference(t)) => t,
        _ => panic!("expected detect inference"),
    };
    let v: Vec<f32> = Vec::try_from(&out.reshape([-1])).unwrap();
    let expect: Vec<f64> = js["values"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    assert_eq!(out.numel(), 1 * 84 * 8400, "output shape mismatch");
    let v = &v[..expect.len()]; // 参考只存前 4096 个值
    let mut max_diff = 0f64;
    for (a, b) in v.iter().zip(expect.iter()) {
        max_diff = max_diff.max((*a as f64 - *b).abs());
    }
    println!("forward parity max_diff = {max_diff:.3e} over {} values", v.len());
    assert!(max_diff < 1e-3, "forward parity failed: max_diff={max_diff}");
}
