import torch
from accelerate import init_empty_weights

def get_device():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "cuda"
    return "cpu"

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

if __name__ == "__main__":
    print(clean_memory())