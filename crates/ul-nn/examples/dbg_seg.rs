fn main() {
    for (name, pt) in [("yolov8n-seg", "yolov8n-seg.pt"), ("yolov8n-pose", "yolov8n-pose.pt"), ("yolov8n-obb", "yolov8n-obb.pt")] {
        let (sd, _) = ul_nn::pt::load_pt(std::path::Path::new(&format!("weights/{pt}"))).unwrap();
        let mut m = ul_nn::model::YoloModel::build(name, None, tch::Device::Cpu, false).unwrap();
        let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
        println!("{name}: loaded={} sm={} miss={} unexp={}", rep.loaded, rep.shape_mismatch.len(), rep.missing.len(), rep.unexpected.len());
        for n in rep.shape_mismatch.iter().take(3) { println!("  sm: {n}"); }
        for n in rep.unexpected.iter().take(3) { println!("  unexp: {n}"); }
    }
}
