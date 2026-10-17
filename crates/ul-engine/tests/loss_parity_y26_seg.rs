//! yolo26-seg E2E 损失对拍官方（v8SegmentationLoss 双分支 + seg 分量）。
//!
//! 参照由 `python scripts/yolo26_seg_loss_ref.py`（Miniconda 官方 env）生成：
//! yolo26n-seg.pt 完整权重（nc=80）+ 固定输入/标签/实例掩码/语义掩码。
//! sem 分量 port 未实现（官方 sem=1.648，BCEDice），不参与断言。

use std::path::Path;

use tch::{Device, Tensor};

use ul_engine::loss::{detect_branch_loss, segmentation_mask_loss, DetBatch, LossW};
use ul_nn::model::{ModelOutput, YoloModel};
use ul_nn::tal::TaskAlignedAssigner;

#[test]
fn yolo26_seg_e2e_loss_matches_official() {
    let Ok(text) = std::fs::read_to_string("../../weights/yolo26n_seg_loss_ref.json") else {
        eprintln!("skip: weights/yolo26n_seg_loss_ref.json 不存在（python scripts/yolo26_seg_loss_ref.py 生成）");
        return;
    };
    let v: serde_json::Value = serde_json::from_str(&text).expect("parse seg loss ref");
    let (b, s) = (v["batch_size"].as_i64().unwrap(), v["imgsz"].as_i64().unwrap());
    let sq = (s * s) as usize;

    let fvec = |key: &str| -> Vec<f32> {
        v[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
    };
    let img = Tensor::from_slice(&fvec("img")).reshape([b, 3, s, s]);
    let (bidx, cls, bboxes) = (fvec("batch_idx"), fvec("cls"), fvec("bboxes"));
    let n = bidx.len();
    let mut labels: Vec<f32> = Vec::with_capacity(n * 6);
    for i in 0..n {
        labels.extend_from_slice(&[bidx[i], cls[i], bboxes[i * 4], bboxes[i * 4 + 1], bboxes[i * 4 + 2], bboxes[i * 4 + 3]]);
    }
    let labels = Tensor::from_slice(&labels).reshape([n as i64, 6]);

    // port 格式 gt_masks：(B, S, S) f32，值 = 图内实例序号（1 起）——参照 JSON 即此格式
    let gt_masks = Tensor::from_slice(&fvec("masks")).reshape([b, s, s]);

    // 官方同款完整权重（nc=80）
    let (sd, _) = ul_nn::pt::load_pt(Path::new("../../weights/yolo26n-seg.pt")).unwrap();
    let mut m = YoloModel::build("yolo26n-seg", None, Device::Cpu, false).unwrap();
    let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    if !rep.shape_mismatch.is_empty() { panic!("y26-seg 加载有缺口：{:?}
{}", rep.shape_mismatch, rep.summary()); }

    let mut batch = DetBatch::detect(img, labels, b, s);
    batch.masks = Some(gt_masks);
    let out = m.forward(&batch.img, true).unwrap();
    let (a, b_o2o) = match &out {
        ModelOutput::Detect(ul_nn::head::HeadOutput::TrainDual(a, b)) => (a.as_ref(), b.as_ref()),
        _ => panic!("expect TrainDual"),
    };
    let proto = a.proto.as_ref().expect("seg proto");

    let assigner = TaskAlignedAssigner { topk: 10, topk2: 10, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    let assigner_o2o = TaskAlignedAssigner { topk: 7, topk2: 1, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    let (points, strides) = ul_nn::ops::make_anchors(&a.feats, &m.stride, 0.5);

    // o2m：detect 三分量 + seg
    let (_lo, po, ao) = detect_branch_loss(&m, &assigner, a, &batch, &points, &strides, LossW::default());
    let seg_o = f64::try_from(&segmentation_mask_loss(
        &ao, proto, a.extra.as_ref().expect("seg coeffs"), &batch.masks.as_ref().unwrap(), s,
    ) * 7.5)
    .unwrap_or(f64::NAN);
    println!("[o2m] box={:.4} seg={:.4} cls={:.4} l1={:.4}", po.box_, seg_o, po.cls, po.dfl);

    // o2o：detect 三分量 + seg
    let (_lt, pt, at) = detect_branch_loss(&m, &assigner_o2o, b_o2o, &batch, &points, &strides, LossW::default());
    let seg_t = f64::try_from(&segmentation_mask_loss(
        &at, proto, b_o2o.extra.as_ref().expect("seg coeffs"), &batch.masks.as_ref().unwrap(), s,
    ) * 7.5)
    .unwrap_or(f64::NAN);
    println!("[o2o] box={:.4} seg={:.4} cls={:.4} l1={:.4}", pt.box_, seg_t, pt.cls, pt.dfl);

    let ref_o2m = |k: &str| v["o2m"][k].as_f64().unwrap();
    let ref_o2o = |k: &str| v["o2o"][k].as_f64().unwrap();
    let close = |a: f64, r: f64| (a - r).abs() < 5e-3 * r.abs().max(1.0);
    assert!(close(po.box_, ref_o2m("box_loss")), "o2m box: {} vs {}", po.box_, ref_o2m("box_loss"));
    assert!(close(seg_o, ref_o2m("seg_loss")), "o2m seg: {} vs {}", seg_o, ref_o2m("seg_loss"));
    assert!(close(po.cls, ref_o2m("cls_loss")), "o2m cls: {} vs {}", po.cls, ref_o2m("cls_loss"));
    assert!(close(po.dfl, ref_o2m("l1_loss")), "o2m l1: {} vs {}", po.dfl, ref_o2m("l1_loss"));
    assert!(close(pt.box_, ref_o2o("box_loss")), "o2o box: {} vs {}", pt.box_, ref_o2o("box_loss"));
    assert!(close(seg_t, ref_o2o("seg_loss")), "o2o seg: {} vs {}", seg_t, ref_o2o("seg_loss"));
    assert!(close(pt.cls, ref_o2o("cls_loss")), "o2o cls: {} vs {}", pt.cls, ref_o2o("cls_loss"));
    assert!(close(pt.dfl, ref_o2o("l1_loss")), "o2o l1: {} vs {}", pt.dfl, ref_o2o("l1_loss"));
}
