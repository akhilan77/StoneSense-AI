
from pathlib import Path
from getpass import getpass
from huggingface_hub import snapshot_download

model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
target = Path("dl/models/ct/dinov3/backbone")

print("Starting DINOv3 backbone download...")
token = getpass("Enter Hugging Face READ token: ")

target.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=str(target),
    token=token,
)

print(f"Download completed: {target.resolve()}")
