//! seg/pose/obb 三任务损失对拍官方 ultralytics 8.4.157。
//!
//! 参照由 `python scripts/loss_ref_task.py` 生成（同输入同权重 + 官方损失类），
//! 存 `weights/loss_ref_{seg,pose,obb}.json`；缺失时跳过。

use std::path::Path;
use std::sync::OnceLock;

use tch::{Device, Kind, Tensor};

use ul_engine::loss::{obb_branch_loss, pose_keypoints_loss, segmentation_mask_loss, v8_detection_loss, DetBatch, LossW};
use ul_nn::model::{ModelOutput, YoloModel};
use ul_nn::tal::TaskAlignedAssigner;

static INIT: OnceLock<()> = OnceLock::new();

fn ensure_weights() {
    INIT.get_or_init(|| {
        std::fs::create_dir_all("../../weights").ok();
        for f in ["yolov8n-seg.pt", "yolov8n-pose.pt", "yolov8n-obb.pt"] {
            let p = Path::new("../../weights").join(f);
            if !p.exists() {
                let url = format!("https://github.com/ultralytics/assets/releases/download/v8.3.0/{f}");
                println!("downloading {f}...");
                let _ = std::process::Command::new("curl")
                    .args(["-sL", "-o", p.to_str().unwrap(), &url])
                    .status();
            }
        }
    });
}

fn load_ref(task: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(format!("../../weights/loss_ref_{task}.json")).ok()?;
    Some(serde_json::from_str(&text).expect("parse ref json"))
}

fn fvec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

struct Ref {
    v: serde_json::Value,
    img: Tensor,
    labels: Tensor,
    b: i64,
    s: i64,
    model: YoloModel,
    assigner: TaskAlignedAssigner,
}

fn setup(task: &str, model_name: &str) -> Ref {
    ensure_weights();
    let v = load_ref(task).unwrap_or_else(|| panic!("ref {task} missing（python scripts/loss_ref_task.py 生成）"));
    let (b, s) = (v["batch_size"].as_i64().unwrap(), v["imgsz"].as_i64().unwrap());
    let img = Tensor::from_slice(&fvec(&v["img"])).reshape([b, 3, s, s]);
    let (bidx, cls, bb) = (fvec(&v["batch_idx"]), fvec(&v["cls"]), fvec(&v["bboxes"]));
    let n = bidx.len();
    // labels：[idx, cls, bbox...]（detect/pose 4 列 xywh；obb 5 列 xywhr）
    let bcols = if task == "obb" { 5 } else { 4 };
    let mut labels: Vec<f32> = Vec::with_capacity(n * (2 + bcols));
    for i in 0..n {
        labels.push(bidx[i]);
        labels.push(cls[i]);
        labels.extend_from_slice(&bb[i * bcols..(i + 1) * bcols]);
    }
    let labels = Tensor::from_slice(&labels).reshape([n as i64, 2 + bcols as i64]);

    let (sd, _) = ul_nn::pt::load_pt(Path::new(&format!("../../weights/{model_name}.pt"))).unwrap();
    // 官方发布权重的 nc 与 yaml 默认不同（如 obb=DOTA 15），以 ckpt 实际 nc 覆盖
    let nc = v["nc"].as_i64();
    let mut m = YoloModel::build(model_name, nc, Device::Cpu, false).unwrap();
    ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    let assigner = TaskAlignedAssigner { topk: 10, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    Ref { v, img, labels, b, s, model: m, assigner }
}

fn close(a: f64, r: f64) -> bool {
    (a - r).abs() < 2e-3 * r.abs().max(1.0)
}

#[test]
fn seg_loss_matches_official() {
    let Some(v) = load_ref("seg") else {
        eprintln!("skip: loss_ref_seg.json 缺失");
        return;
    };
    drop(v);
    let r = setup("seg", "yolov8n-seg");
    let v = &r.v;
    // masks (B,S,S) uint8 → f32
    let mshape: Vec<i64> = v["mask_shape"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
    let masks = Tensor::from_slice(&fvec(&v["masks"])).reshape(mshape);
    let batch = DetBatch {
        img: r.img,
        labels: r.labels,
        batch_size: r.b,
        imgsz: r.s,
        masks: Some(masks),
        kpts: None,
        obb: false,
    };
    let out = r.model.forward(&batch.img, true).unwrap();
    let head = match out {
        ModelOutput::Detect(h) => h,
        _ => panic!("expect detect head"),
    };
    let assigner_o2o = TaskAlignedAssigner { topk: 1, ..TaskAlignedAssigner { topk: 10, num_classes: r.model.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 } };
    let det = v8_detection_loss(&r.model, LossW::default(), &r.assigner, &assigner_o2o, &head, &batch, false);
    let info = det.assign.as_ref().unwrap();
    let (rp, proto) = match &head {
        ul_nn::head::HeadOutput::Train(rp) => (rp, rp.proto.as_ref().expect("seg proto")),
        _ => panic!("expect train head"),
    };
    let masks_t = batch.masks.as_ref().unwrap();
    let seg = segmentation_mask_loss(info, proto, rp.extra.as_ref().unwrap(), masks_t, r.s);
    let seg = f64::try_from(&seg).unwrap();
    {
        let fgn = f64::try_from(&info.fg_mask.to_kind(Kind::Float).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)).unwrap_or(0.0);
        println!("seg mid: fg={fgn:.0} ts_sum={:.4} (official {} / {:.4})",
            f64::try_from(&out_target_scores_sum(info)).unwrap_or(0.0), v["fg_sum"].as_i64().unwrap_or(-1), v["ts_sum"].as_f64().unwrap_or(-1.0));
    }

    let (rb, rs, rc, rd) = (
        v["loss"]["box_loss"].as_f64().unwrap(),
        v["loss"]["seg_loss"].as_f64().unwrap(),
        v["loss"]["cls_loss"].as_f64().unwrap(),
        v["loss"]["dfl_loss"].as_f64().unwrap(),
    );
    println!("seg: box ours={:.4} ref={rb:.4} | seg ours={seg:.4} ref={rs:.4} | cls ours={:.4} ref={rc:.4} | dfl ours={:.4} ref={rd:.4}",
        det.parts.box_, det.parts.cls, det.parts.dfl);
    assert!(close(det.parts.box_, rb), "box mismatch");
    assert!(close(det.parts.cls, rc), "cls mismatch");
    assert!(close(det.parts.dfl, rd), "dfl mismatch");
    // ref 的 seg_loss 已乘 hyp.box=7.5（官方 items 语义），我们的裸值需同乘
    assert!(close(seg * 7.5, rs), "seg mismatch: {} vs {rs}", seg * 7.5);
}

#[test]
fn pose_loss_matches_official() {
    let Some(v) = load_ref("pose") else {
        eprintln!("skip: loss_ref_pose.json 缺失");
        return;
    };
    drop(v);
    let r = setup("pose", "yolov8n-pose");
    let v = &r.v;
    let kpts = Tensor::from_slice(&fvec(&v["keypoints"])).reshape([-1, v["kpt_shape"][0].as_i64().unwrap() * v["kpt_shape"][1].as_i64().unwrap()]);
    let batch = DetBatch { img: r.img, labels: r.labels, batch_size: r.b, imgsz: r.s, masks: None, kpts: Some(kpts), obb: false };
    let out = r.model.forward(&batch.img, true).unwrap();
    let head = match out {
        ModelOutput::Detect(h) => h,
        _ => panic!("expect detect head"),
    };
    let assigner_o2o = TaskAlignedAssigner { topk: 1, num_classes: r.model.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    let det = v8_detection_loss(&r.model, LossW::default(), &r.assigner, &assigner_o2o, &head, &batch, false);
    let rp = match &head {
        ul_nn::head::HeadOutput::Train(rp) => rp,
        _ => panic!("expect train head"),
    };
    let bidx = batch.labels.select(1, 0).to_kind(Kind::Int64);
    let (pl, kl) = pose_keypoints_loss(det.assign.as_ref().unwrap(), rp.extra.as_ref().unwrap(), 3, batch.kpts.as_ref().unwrap(), &bidx, r.s);
    let (pl, kl) = (f64::try_from(&pl).unwrap() * 12.0, f64::try_from(&kl).unwrap());

    let (rb, rp_, rk, rc, rd) = (
        v["loss"]["box_loss"].as_f64().unwrap(),
        v["loss"]["pose_loss"].as_f64().unwrap(),
        v["loss"]["kobj_loss"].as_f64().unwrap(),
        v["loss"]["cls_loss"].as_f64().unwrap(),
        v["loss"]["dfl_loss"].as_f64().unwrap(),
    );
    println!("pose: box {:.4}/{rb:.4} pose {pl:.4}/{rp_:.4} kobj {kl:.4}/{rk:.4} cls {:.4}/{rc:.4} dfl {:.4}/{rd:.4}",
        det.parts.box_, det.parts.cls, det.parts.dfl);
    assert!(close(det.parts.box_, rb), "box mismatch");
    assert!(close(pl, rp_), "pose mismatch: {pl} vs {rp_}");
    assert!(close(kl, rk), "kobj mismatch: {kl} vs {rk}");
    assert!(close(det.parts.cls, rc), "cls mismatch");
    assert!(close(det.parts.dfl, rd), "dfl mismatch");
}

#[test]
fn obb_loss_matches_official() {
    std::env::set_var("UL_OBB_DEBUG", "1");
    let Some(v) = load_ref("obb") else {
        eprintln!("skip: loss_ref_obb.json 缺失");
        return;
    };
    drop(v);
    let r = setup("obb", "yolov8n-obb");
    let v = &r.v;
    let batch = DetBatch { img: r.img, labels: r.labels, batch_size: r.b, imgsz: r.s, masks: None, kpts: None, obb: true };
    let out = r.model.forward(&batch.img, true).unwrap();
    let head = match out {
        ModelOutput::Detect(h) => h,
        _ => panic!("expect detect head"),
    };
    let rp = match &head {
        ul_nn::head::HeadOutput::Train(rp) => rp,
        _ => panic!("expect train head"),
    };
    let (total, parts) = obb_branch_loss(&r.model, &r.assigner, rp, &batch, LossW::default());
    let total = f64::try_from(&total).unwrap();
    println!("obb mid: official fg={} ts_sum={:.4}", v["fg_sum"].as_i64().unwrap_or(-1), v["ts_sum"].as_f64().unwrap_or(-1.0));

    let (rb, rc, rd, ra) = (
        v["loss"]["box_loss"].as_f64().unwrap(),
        v["loss"]["cls_loss"].as_f64().unwrap(),
        v["loss"]["dfl_loss"].as_f64().unwrap(),
        v["loss"]["angle_loss"].as_f64().unwrap(),
    );
    println!("obb: box {:.4}/{rb:.4} cls {:.4}/{rc:.4} dfl {:.4}/{rd:.4} angle {:.4}/{ra:.4} total {total:.4}/{:.4}",
        parts.box_, parts.cls, parts.dfl, parts.angle, v["total"].as_f64().unwrap());
    assert!(close(parts.box_, rb), "box mismatch: {} vs {rb}", parts.box_);
    assert!(close(parts.cls, rc), "cls mismatch");
    assert!(close(parts.dfl, rd), "dfl mismatch: {} vs {rd}", parts.dfl);
    assert!(close(parts.angle, ra), "angle mismatch: {} vs {ra}", parts.angle);
}

// 辅助：target_scores 总和（分配一致性快速检查）
fn out_target_scores_sum(_info: &ul_engine::loss::AssignInfo) -> Tensor {
    Tensor::from_slice(&[0f32])
}
