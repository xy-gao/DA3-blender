import glob, os, torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
MODEL_DIR = "./model"
MODEL_PATH = "./model/model.safetensors"
# torch.hub.download_url_to_file("https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors", MODEL_PATH)
model = DepthAnything3()
from safetensors.torch import load_file
weight = load_file(MODEL_PATH)
model.load_state_dict(weight, strict=False)
model = model.to(device=device)
example_path = "examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(
    images,
)