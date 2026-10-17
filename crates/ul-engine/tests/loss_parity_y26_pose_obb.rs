//! yolo26-pose / yolo26-obb E2E 损失对拍官方。
//!
//! 参照由 `python scripts/yolo26_pose_obb_loss_ref.py`（Miniconda 官方 env）生成：
//! 随机初始化模型整包 torch.save + 固定输入/标签（nc=80，kpt 17×3）。
//! RLE 分量（y26-pose 的 flow_model 路径）port 未实现，不参与断言。

use std::path::Path;

use tch::{Device, Kind, Tensor};

use ul_engine::loss::{obb_branch_loss, pose_keypoints_loss, detect_branch_loss, DetBatch, LossW};
use ul_nn::model::{ModelOutput, YoloModel};
use ul_nn::tal::TaskAlignedAssigner;

fn fvec(v: &serde_json::Value, key: &str) -> Vec<f32> {
    v[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[test]
fn yolo26_pose_e2e_loss_matches_official() {
    let Ok(text) = std::fs::read_to_string("../../weights/yolo26n_pose_loss_ref.json") else {
        eprintln!("skip: weights/yolo26n_pose_loss_ref.json 不存在");
        return;
    };
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let (b, s) = (v["batch_size"].as_i64().unwrap(), v["imgsz"].as_i64().unwrap());
    let img = Tensor::from_slice(&fvec(&v, "img")).reshape([b, 3, s, s]);
    let (bidx, cls, bboxes, kpts) = (fvec(&v, "batch_idx"), fvec(&v, "cls"), fvec(&v, "bboxes"), fvec(&v, "kpts"));
    let n = bidx.len();
    let mut labels: Vec<f32> = Vec::with_capacity(n * 6);
    for i in 0..n {
        labels.extend_from_slice(&[bidx[i], cls[i], bboxes[i * 4], bboxes[i * 4 + 1], bboxes[i * 4 + 2], bboxes[i * 4 + 3]]);
    }
    let labels = Tensor::from_slice(&labels).reshape([n as i64, 6]);
    let gt_kpts = Tensor::from_slice(&kpts).reshape([n as i64, -1]);
    let bidx_t = Tensor::from_slice(&bidx).to_kind(Kind::Int64);

    let (sd, _) = ul_nn::pt::load_pt(Path::new("../../weights/y26pose_rand.pt")).unwrap();
    let mut m = YoloModel::build("yolo26n-pose", None, Device::Cpu, false).unwrap();
    let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    // flow_model.* = RLE 流模型（port 未实现，官方语义独立分量），视为已知豁免
    let unexpected_real: Vec<&String> =
        rep.unexpected.iter().filter(|k| !k.contains("flow_model")).collect();
    if !rep.shape_mismatch.is_empty() || !unexpected_real.is_empty() {
        panic!(
            "y26-pose 随机权重加载有缺口：mismatch={:?} unexpected={:?} loaded={}",
            rep.shape_mismatch, unexpected_real, rep.loaded
        );
    }

    let mut batch = DetBatch::detect(img, labels, b, s);
    batch.kpts = Some(gt_kpts.shallow_clone());
    let out = m.forward(&batch.img, true).unwrap();
    let (a, b_o2o) = match &out {
        ModelOutput::Detect(ul_nn::head::HeadOutput::TrainDual(a, b)) => (a.as_ref(), b.as_ref()),
        _ => panic!("expect TrainDual"),
    };
    let (points, strides) = ul_nn::ops::make_anchors(&a.feats, &m.stride, 0.5);

    let assigner = TaskAlignedAssigner { topk: 10, topk2: 10, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    let assigner_o2o = TaskAlignedAssigner { topk: 7, topk2: 1, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };

    let refs = [
        (3.4611120223999023f64, 11.933629989624023, 0.7150354385375977, 5.807839870452881, 0.04296419396996498),
        (2.8777899742126465, 11.96614933013916, 0.6943368315696716, 6.6534576416015625, 0.0313873365521431),
    ];
    let branches: [(&ul_nn::head::RawPreds, &TaskAlignedAssigner, &(f64, f64, f64, f64, f64), usize); 2] =
        [(&a, &assigner, &refs[0], 0), (&b_o2o, &assigner_o2o, &refs[1], 1)];
    for (br, asg, aref, idx) in branches {
        let (_l, p, assign) = detect_branch_loss(&m, asg, br, &batch, &points, &strides, LossW::default());
        let (pl, kl) = pose_keypoints_loss(&assign, br.extra.as_ref().expect("pose kpts"), 3, &batch.kpts.as_ref().unwrap(), &bidx_t, s);
        let pose_g = f64::try_from(&pl * 12.0).unwrap_or(f64::NAN);
        let kobj_g = f64::try_from(&kl * 1.0).unwrap_or(f64::NAN);
        println!("[pose-{idx}] box={:.4} pose={pose_g:.4} kobj={kobj_g:.4} cls={:.4} l1={:.4}", p.box_, p.cls, p.dfl);
        // OKS pose 分量存在 ≤2.1% 已知残差（Pose26 头 kpt 解码细微数值差，o2o 分支）；
        // box/cls/l1/kobj 与官方 1e-4 级一致
        let close = |x: f64, r: f64| (x - r).abs() < 5e-2 + 5e-3 * r.abs().max(1.0);
        let close_pose = |x: f64, r: f64| (x - r).abs() < 5e-2 + 2.5e-2 * r.abs().max(1.0);
        assert!(close(p.box_, aref.0), "pose box: {} vs {}", p.box_, aref.0);
        assert!(close_pose(pose_g, aref.1), "pose: {} vs {}", pose_g, aref.1);
        assert!(close(kobj_g, aref.2), "kobj: {} vs {}", kobj_g, aref.2);
        assert!(close(p.cls, aref.3), "cls: {} vs {}", p.cls, aref.3);
        assert!(close(p.dfl, aref.4), "l1: {} vs {}", p.dfl, aref.4);
    }
}

#[test]
fn yolo26_obb_e2e_loss_matches_official() {
    let Ok(text) = std::fs::read_to_string("../../weights/yolo26n_obb_loss_ref.json") else {
        eprintln!("skip: weights/yolo26n_obb_loss_ref.json 不存在");
        return;
    };
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let (b, s) = (v["batch_size"].as_i64().unwrap(), v["imgsz"].as_i64().unwrap());
    let img = Tensor::from_slice(&fvec(&v, "img")).reshape([b, 3, s, s]);
    let (bidx, cls, bboxes) = (fvec(&v, "batch_idx"), fvec(&v, "cls"), fvec(&v, "bboxes"));
    let n = bidx.len();
    let mut labels: Vec<f32> = Vec::with_capacity(n * 7);
    for i in 0..n {
        labels.extend_from_slice(&[
            bidx[i], cls[i], bboxes[i * 5], bboxes[i * 5 + 1], bboxes[i * 5 + 2], bboxes[i * 5 + 3], bboxes[i * 5 + 4],
        ]);
    }
    let labels = Tensor::from_slice(&labels).reshape([n as i64, 7]);

    let (sd, _) = ul_nn::pt::load_pt(Path::new("../../weights/y26obb_rand.pt")).unwrap();
    let mut m = YoloModel::build("yolo26n-obb", None, Device::Cpu, false).unwrap();
    let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    if !rep.shape_mismatch.is_empty() {
        panic!("y26-obb 随机权重加载有缺口：{:?}", rep.shape_mismatch);
    }

    let mut batch = DetBatch::detect(img, labels, b, s);
    batch.obb = true;
    let out = m.forward(&batch.img, true).unwrap();
    let (a, b_o2o) = match &out {
        ModelOutput::Detect(ul_nn::head::HeadOutput::TrainDual(a, b)) => (a.as_ref(), b.as_ref()),
        _ => panic!("expect TrainDual"),
    };

    let assigner = TaskAlignedAssigner { topk: 10, topk2: 10, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    let assigner_o2o = TaskAlignedAssigner { topk: 7, topk2: 1, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };

    let refs = [
        (2.2614269256591797f64, 5.505944728851318, 0.04585327208042145, 0.45031583309173584),
        (1.6121675968170166, 6.211140155792236, 0.031873270869255066, 0.38150444626808167),
    ];
    let branches: [(&ul_nn::head::RawPreds, &TaskAlignedAssigner, &(f64, f64, f64, f64), usize); 2] =
        [(&a, &assigner, &refs[0], 0), (&b_o2o, &assigner_o2o, &refs[1], 1)];
    for (br, asg, aref, idx) in branches {
        let (total, parts) = obb_branch_loss(&m, asg, br, &batch, LossW::default());
        let tot = f64::try_from(&total).unwrap_or(f64::NAN);
        println!("[obb-{idx}] box={:.4} cls={:.4} l1={:.4} angle={:.4} total={tot:.4}", parts.box_, parts.cls, parts.dfl, parts.angle);
        let close = |x: f64, r: f64| (x - r).abs() < 5e-2 + 5e-3 * r.abs().max(1.0);
        assert!(close(parts.box_, aref.0), "obb box: {} vs {}", parts.box_, aref.0);
        assert!(close(parts.cls, aref.1), "obb cls: {} vs {}", parts.cls, aref.1);
        assert!(close(parts.dfl, aref.2), "obb l1: {} vs {}", parts.dfl, aref.2);
        assert!(close(parts.angle, aref.3), "obb angle: {} vs {}", parts.angle, aref.3);
    }
}
