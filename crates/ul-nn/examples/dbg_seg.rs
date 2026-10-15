fn main() {
    let (sd, _) = ul_nn::pt::load_pt(std::path::Path::new("weights/yolov8n-obb.pt")).unwrap();
    let mut m = ul_nn::model::YoloModel::build("yolov8n-obb", None, tch::Device::Cpu, false).unwrap();
    let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    for n in rep.shape_mismatch.iter().take(6) {
        println!("sm: {}", n);
    }
    println!("nc={} reg_max={}", m.nc, m.reg_max);
    if let Some(ul_nn::parse::Module::Head(ul_nn::parse::HeadMod::Detect(dh))) = m.layers.last().map(|l| &l.module) {
        println!("cv3[0] s0: {:?}", match &dh.cv3[0].stage0 {
            ul_nn::head::BranchStage::Conv(c) => c.conv.weight.size(),
            _ => vec![0],
        });
    }
}
