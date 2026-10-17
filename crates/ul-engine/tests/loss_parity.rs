//! v8DetectionLoss 三分量对拍官方 ultralytics 8.4.157。
//!
//! 参照由 `python scripts/loss_ref.py` 生成（固定输入/标签 + 官方前向），存
//! `weights/yolov8n_loss_ref.json`；缺失时跳过（离线环境仍可跑其余测试）。

use std::path::Path;
use std::sync::OnceLock;

use tch::{Device, Tensor};

use ul_engine::loss::{v8_detection_loss, DetBatch, LossW};
use ul_nn::model::{ModelOutput, YoloModel};
use ul_nn::tal::TaskAlignedAssigner;

static INIT: OnceLock<()> = OnceLock::new();

fn ensure_ref() -> Option<serde_json::Value> {
    INIT.get_or_init(|| {
        if !Path::new("../../weights/yolov8n.pt").exists() {
            std::fs::create_dir_all("../../weights").ok();
            let url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt";
            println!("downloading yolov8n.pt...");
            let _ = std::process::Command::new("curl")
                .args(["-sL", "-o", "../../weights/yolov8n.pt", url])
                .status();
        }
    });
    let text = std::fs::read_to_string("../../weights/yolov8n_loss_ref.json").ok()?;
    Some(serde_json::from_str(&text).expect("parse loss ref json"))
}

fn fvec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[test]
fn loss_components_match_official() {
    let Some(v) = ensure_ref() else {
        eprintln!("skip: weights/yolov8n_loss_ref.json 不存在（python scripts/loss_ref.py 生成）");
        return;
    };
    let (b, s) = (v["batch_size"].as_i64().unwrap(), v["imgsz"].as_i64().unwrap());

    let img = Tensor::from_slice(&fvec(&v["img"])).reshape([b, 3, s, s]);
    let (bidx, cls, bboxes) = (fvec(&v["batch_idx"]), fvec(&v["cls"]), fvec(&v["bboxes"]));
    let n = bidx.len();
    let mut labels: Vec<f32> = Vec::with_capacity(n * 6);
    for i in 0..n {
        labels.extend_from_slice(&[bidx[i], cls[i], bboxes[i * 4], bboxes[i * 4 + 1], bboxes[i * 4 + 2], bboxes[i * 4 + 3]]);
    }
    let labels = Tensor::from_slice(&labels).reshape([n as i64, 6]);

    // 官方同款权重 + train 模式前向（BN 用批统计，与官方 net.train() 一致）
    let (sd, _) = ul_nn::pt::load_pt(Path::new("../../weights/yolov8n.pt")).unwrap();
    let mut m = YoloModel::build("yolov8n", None, Device::Cpu, false).unwrap();
    ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();

    let batch = DetBatch::detect(img, labels, b, s);
    let out = m.forward(&batch.img, true).unwrap();
    let head = match out {
        ModelOutput::Detect(h) => h,
        _ => panic!("expect detect head output"),
    };
    let assigner = TaskAlignedAssigner { topk: 10, topk2: 10, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    let assigner_o2o = TaskAlignedAssigner { topk: 1, topk2: 1, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    let det = v8_detection_loss(&m, LossW::default(), &assigner, &assigner_o2o, &head, &batch, false, 1.0, 0.0);

    let (rk, rc, rd) = (
        v["loss"]["box_loss"].as_f64().unwrap(),
        v["loss"]["cls_loss"].as_f64().unwrap(),
        v["loss"]["dfl_loss"].as_f64().unwrap(),
    );
    println!("box  ours={:.6} ref={rk:.6} diff={:+.2e}", det.parts.box_, det.parts.box_ - rk);
    println!("cls  ours={:.6} ref={rc:.6} diff={:+.2e}", det.parts.cls, det.parts.cls - rc);
    println!("dfl  ours={:.6} ref={rd:.6} diff={:+.2e}", det.parts.dfl, det.parts.dfl - rd);
    println!("fg   ours={} ts_sum={:.6}", v["fg_count"].as_i64().unwrap_or(-1), v["ts_sum"].as_f64().unwrap_or(-1.0));

    let close = |a: f64, r: f64| (a - r).abs() < 2e-3 * r.abs().max(1.0);
    assert!(close(det.parts.box_, rk), "box mismatch: {} vs {rk}", det.parts.box_);
    assert!(close(det.parts.cls, rc), "cls mismatch: {} vs {rc}", det.parts.cls);
    assert!(close(det.parts.dfl, rd), "dfl mismatch: {} vs {rd}", det.parts.dfl);
    let total = f64::try_from(&det.total).unwrap();
    println!("total ours={total:.6} ref={:.6}", v["total"].as_f64().unwrap());
}
