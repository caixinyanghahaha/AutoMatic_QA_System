import os
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from bitsandbytes import __version__ as bnb_version


def test_device():
    """测试设备是否可以使用GPU CUDA进行运算"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cpu count: {os.cpu_count()}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"bitsandbytes版本: {bnb_version}")

    # 创建一个张量并移动到GPU
    x = torch.rand(5, 3)
    if torch.cuda.is_available():
        x = x.to('cuda')
        print(x)

def get_device():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("mps")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("cuda")
    else:
        print("cpu")

def clean_memory():
    """清理GPU/MPS内存"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model_safely(model_class, model_name, device):
    """安全加载大模型"""
    with init_empty_weights():
        model = model_class.from_pretrained(model_name)

    # 根据设备选择精度
    if device == "mps":
        return model.to(device, dtype=torch.bfloat16)
    return model.to(device)

def download_model():
    snapshot_download(
        repo_id="deepseek-ai/deepseek-math-7b-instruct",
        local_dir="../local_models/deepseek-math-7b-instruct",
        revision="main"
    )

if __name__ == "__main__":
    test_device()