//! yolo26 E2E 损失对拍官方 ultralytics（reg_max=1 + one2many/one2one 双分支）。
//!
//! 参照由 `python scripts/yolo26_loss_ref.py`（Miniconda 官方 env）生成：
//! 固定输入/标签 + yolo26n.pt 完整权重（nc=80），存 weights/yolo26n_loss_ref.json；
//! 缺失时跳过（离线环境仍可跑其余测试）。
//!
//! 这是 yolo26 训练劣化排查的对拍基线：此前项目的损失对拍只覆盖 v8 的
//! reg_max=16 路径，reg_max=1 的归一化 L1 + E2E 双分支从未与官方对过。

use std::path::Path;

use tch::{Device, Tensor};

use ul_engine::loss::{v8_detection_loss, DetBatch, LossW};
use ul_nn::model::{ModelOutput, YoloModel};
use ul_nn::tal::TaskAlignedAssigner;

#[test]
fn yolo26_e2e_loss_matches_official() {
    let Ok(text) = std::fs::read_to_string("../../weights/yolo26n_loss_ref.json") else {
        eprintln!("skip: weights/yolo26n_loss_ref.json 不存在（python scripts/yolo26_loss_ref.py 生成）");
        return;
    };
    let v: serde_json::Value = serde_json::from_str(&text).expect("parse y26 loss ref json");
    let (b, s) = (v["batch_size"].as_i64().unwrap(), v["imgsz"].as_i64().unwrap());

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

    // 官方同款完整权重（nc=80），train 模式前向（BN 批统计与官方 net.train() 一致）
    let (sd, _) = ul_nn::pt::load_pt(Path::new("../../weights/yolo26n.pt")).unwrap();
    let mut m = YoloModel::build("yolo26n", None, Device::Cpu, false).unwrap();
    let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    assert_eq!(rep.shape_mismatch.len(), 0, "yolo26n.pt 加载应零缺口：{}", rep.summary());

    let batch = DetBatch::detect(img, labels, b, s);
    let out = m.forward(&batch.img, true).unwrap();
    let head = match out {
        ModelOutput::Detect(h) => h,
        _ => panic!("expect detect head output"),
    };
    assert!(m.end2end, "yolo26 应为 end2end 双分支");

    let assigner = TaskAlignedAssigner { topk: 10, topk2: 10, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    let assigner_o2o = TaskAlignedAssigner { topk: 7, topk2: 1, num_classes: m.nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };

    // 逐分支对拍（官方诊断值：o2m fg=44 items=[3.771, 7.094, 0.0692]；
    //            o2o(topk7/topk2=1) fg=5 items=[3.562, 8.329, 0.0486]）
    {
        let (o2m, o2o) = match head {
            ul_nn::head::HeadOutput::TrainDual(ref a, ref b) => (a.as_ref(), b.as_ref()),
            _ => panic!("expect TrainDual"),
        };
        let (points, strides) = ul_nn::ops::make_anchors(&o2m.feats, &m.stride, 0.5);
        let (_lo, po, ao) =
            ul_engine::loss::detect_branch_loss(&m, &assigner, o2m, &batch, &points, &strides, LossW::default());
        let fg_o = f64::try_from(&ao.fg_mask.to_kind(tch::Kind::Float).sum_dim_intlist(None::<&[i64]>, false, tch::Kind::Float)).unwrap_or(-1.0);
        println!("[o2m] box={:.4} cls={:.4} l1={:.4} fg={}", po.box_, po.cls, po.dfl, fg_o);
        let (_lt, pt, at) =
            ul_engine::loss::detect_branch_loss(&m, &assigner_o2o, o2o, &batch, &points, &strides, LossW::default());
        let fg_t = f64::try_from(&at.fg_mask.to_kind(tch::Kind::Float).sum_dim_intlist(None::<&[i64]>, false, tch::Kind::Float)).unwrap_or(-1.0);
        println!("[o2o] box={:.4} cls={:.4} l1={:.4} fg={}", pt.box_, pt.cls, pt.dfl, fg_t);
    }

    let det = v8_detection_loss(&m, LossW::default(), &assigner, &assigner_o2o, &head, &batch, true, 0.8, 0.2);

    // 官方 items = one2one 分支分量（不 ×bs）；loss.sum() = (o2m×0.8 + o2o×0.2) ×bs
    let (rk, rc, rl) = (
        v["loss"]["box_loss"].as_f64().unwrap(),
        v["loss"]["cls_loss"].as_f64().unwrap(),
        v["loss"]["l1_loss"].as_f64().unwrap(),
    );
    println!("box  ours={:.6} ref={rk:.6} diff={:+.2e}", det.parts.box_, det.parts.box_ - rk);
    println!("cls  ours={:.6} ref={rc:.6} diff={:+.2e}", det.parts.cls, det.parts.cls - rc);
    println!("l1   ours={:.6} ref={rl:.6} diff={:+.2e}", det.parts.dfl, det.parts.dfl - rl);
    if let Some(tot) = v["total_sum"].as_f64() {
        let ours_total = f64::try_from(&det.total).unwrap_or(f64::NAN);
        println!("total ours={ours_total:.6} ref={tot:.6}");
        assert!((ours_total - tot).abs() < 5e-3 * tot.abs().max(1.0), "total mismatch: {ours_total} vs {tot}");
    }

    let close = |a: f64, r: f64| (a - r).abs() < 5e-3 * r.abs().max(1.0);
    assert!(close(det.parts.box_, rk), "box mismatch: {} vs {rk}", det.parts.box_);
    assert!(close(det.parts.cls, rc), "cls mismatch: {} vs {rc}", det.parts.cls);
    assert!(close(det.parts.dfl, rl), "l1 mismatch: {} vs {rl}", det.parts.dfl);
}
