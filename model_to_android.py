import os
import torch

src_dir = "infer_models"
dst_dir = "Android/app/src/main/assets"

os.makedirs(dst_dir, exist_ok=True)

for name in ["PNet", "RNet", "ONet"]:
    src = os.path.join(src_dir, f"{name}.pth")
    dst = os.path.join(dst_dir, f"{name}.pt")

    # 关键点：强制映射到 CPU
    model = torch.jit.load(src, map_location="cpu")
    model = model.eval()
    model = torch.jit.freeze(model)

    # 保存为 Android 可加载的 CPU TorchScript
    torch.jit.save(model, dst)
    print(f"导出完成: {dst}")