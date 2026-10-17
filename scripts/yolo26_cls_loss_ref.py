# 官方 yolo26-cls 随机初始化 + 固定输入的 CE 损失参照。
import torch

code = r'''
import torch, json
from ultralytics import YOLO
from types import SimpleNamespace

torch.manual_seed(2026)
m = YOLO("yolo26n-cls.yaml")
net = m.model
if isinstance(net.args, dict):
    net.args = SimpleNamespace(**net.args)
torch.save({"model": net, "train_args": {}}, "E:/ultralytics-rust/weights/y26cls_rand.pt")

torch.manual_seed(SEED := 2026 + 7)
img = torch.rand(2, 3, 64, 64)
targets = torch.tensor([2, 55])
loss, items = net.loss({"img": img, "cls": targets})
logits = net(img)
logits = logits[1] if isinstance(logits, tuple) else logits
print("logits mean/std:", float(logits.mean()), float(logits.std()))
print("cls loss:", float(loss), items)

out = {
    "batch_size": 2, "imgsz": 64,
    "img": img.reshape(-1).tolist(),
    "targets": targets.tolist(),
    "loss": float(loss),
    "logits": logits.detach().reshape(-1).tolist(),
}
with open("E:/ultralytics-rust/weights/yolo26n_cls_loss_ref.json", "w") as f:
    json.dump(out, f)
'''

exec(code)
