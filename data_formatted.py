import json
import re

from transformers import AutoTokenizer

"""
此代码用于将数据集处理为连续对话（DialoGPT格式）。

功能包括：
- 单条长文本，用分隔符连接多轮对话
- 输入输出结构 "T: A[SEP]S: B[SEP]T: C" → 预测下一个词
- 信息利用率 保留完整对话流，适合多轮交互
- 适用场景 开放式对话、长上下文依赖
"""

# ========== 配置部分 ==========
INPUT_FILE = "data/mrbench_v3_devset.json"  # 输入文件路径（需提前准备）
OUTPUT_FILE = "data/dialogpt_formatted.json"  # 输出文件路径
MODEL_NAME = "microsoft/DialoGPT-medium"  # 或替换为其他版本如 'microsoft/DialoGPT-large'
# =============================

# 加载分词器并配置分隔符，添加特殊标记：pad_token: 用于填充序列，确保输入长度一致。sep_token: 连接对话轮次，可以用作结束标记。
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({
    "pad_token": "[PAD]",  # 确保填充标记存在
    "sep_token": "[SEP]"  # 将 sep_token 设为 eos_token（可选）
})

def split_and_clean_conversation(history_str):
    """拆分对话并清洗内容"""
    # 步骤1: 拆分对话轮次（兼容换行符和特殊空格）
    turns = re.split(r'\n\s*|\u00A0|\u2028', history_str.strip())  # \u2028 是行分隔符

    # 步骤2: 合并被错误分割的轮次（如数学推导步骤）
    merged_turns = []
    current_speaker = None
    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue
        # 检测是否为新轮次（如以 "Tutor:" 或 "Student:" 开头）
        if re.match(r'^(Tutor|Student):', turn):
            if current_speaker is not None:
                merged_turns.append(current_turn)
            current_speaker = turn.split(":")[0]
            current_turn = turn
        else:
            # 如果未检测到新说话者，合并到上一轮次
            current_turn += " " + turn
    if current_turn:
        merged_turns.append(current_turn)

    # 步骤3: 清洗内容（删除多余空格、换行）
    cleaned_turns = [
        re.sub(r'\s+', ' ', turn).strip()
        for turn in merged_turns
    ]
    return cleaned_turns


# 加载原始数据（示例结构）
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)  # 假设数据格式为 [{"conversation_history": "..."}]

# 处理所有对话
formatted_data = []
for item in raw_data:
    history_str = item["conversation_history"]
    turns = split_and_clean_conversation(history_str)

    # 转换为 DialoGPT 格式（用 [SEP] 连接轮次）
    dialogpt_text = tokenizer.sep_token.join(turns)

    formatted_data.append({"text": dialogpt_text})

# 保存处理后的数据
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print(f"转换完成！处理后的数据已保存至 {OUTPUT_FILE}")