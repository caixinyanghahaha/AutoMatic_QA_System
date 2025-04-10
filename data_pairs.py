from typing import List, Dict
import json
import re
from transformers import AutoTokenizer

"""
此代码用于将数据集处理为训练的对话对。

功能包括：
- 多条独立样本，每条包含一个上下文和对应回复
- 输入输出结构 "T: A[SEP]S: B" → "T: C"
- 信息利用率 明确分离上下文和目标，适合单轮响应生成
- 适用场景 任务型对话、精确控制回复生成
"""

# 配置部分
INPUT_FILE = "data/mrbench_v3_devset.json"  # 输入文件路径
OUTPUT_FILE = "data/dialog_pairs.json"  # 输出文件路径
MODEL_NAME = "microsoft/DialoGPT-medium"  # 用于分词的模型
MIN_CONTEXT_TURNS = 1  # 上下文最少轮次, 确保生成的对话对至少包含指定数量的历史轮次。这有助于确保上下文足够丰富，以便模型能够理解对话的背景，从而生成更相关的响应。
MAX_HISTORY = 3  # 最大历史轮次数（防止过长）, 限制用于生成上下文的历史轮次数，以防止上下文过长，导致输入超出模型的处理能力。这有助于保持输入的有效性和效率，同时避免模型因上下文过多而产生的性能下降

# 初始化分词器（仅用于分隔符处理）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({
    "pad_token": "[PAD]",  # 确保填充标记存在
    "sep_token": "[SEP]"  # 将 sep_token 设为 eos_token（可选）
})

def clean_text(text: str) -> str:
    """清洗文本，删除多余的空格和特殊字符，保持文本整洁。"""
    text = re.sub(r'\s+', ' ', text)  # 合并多余空格
    text = re.sub(r'(\u00A0|\u2028|\r\n)', ' ', text)  # 替换特殊空白符
    return text.strip()

def split_conversation(history: str) -> List[Dict]:
    """将原始对话字符串拆分为结构化轮次"""
    # 用于存储每一轮对话的信息，最终作为返回值。
    turns = []
    current_speaker = None
    current_text = []
    # 将整个对话按行分割，并逐行处理。
    for line in history.split('\n'):
        # 使用 clean_text 函数清洗每一行，去除多余的空格和特殊字符。如果清洗后的行为空，则跳过该行。
        line = clean_text(line)
        if not line:
            continue
        # 使用正则表达式检测行的开头是否包含说话者的名称（如 "Tutor" 或 "Student"）。
        # 如果 current_speaker 和 current_text 不为空，说明已有一轮对话，保存这一轮的说话者和发言内容到 turns 列表。
        # 更新 current_speaker 和 current_text，将当前说话者的发言内容初始化为当前行的文本。
        # 如果没有检测到说话者，说明这是当前说话者的续发言，将其追加到 current_text 中。
        speaker_match = re.match(r'^(Tutor|Student|老师|学生)[：:]?\s*(.*)', line)
        if speaker_match:
            if current_speaker and current_text:  # 保存上一轮
                turns.append({
                    "speaker": current_speaker,
                    "text": ' '.join(current_text)
                })
            current_speaker = speaker_match.group(1)
            current_text = [speaker_match.group(2)]
        else:
            current_text.append(line)

    # 在循环结束后，检查是否还有未保存的最后一轮对话，如果有，将其添加到 turns 列表中。
    if current_speaker and current_text:
        turns.append({
            "speaker": current_speaker,
            "text": ' '.join(current_text)
        })
    return turns

def generate_pairs(turns: List[Dict]) -> List[Dict]:
    """生成(context, response)训练对，turns 是一个包含多个对话轮次的列表，每个轮次是一个字典，包含说话者和文本。"""
    pairs = [] # 用于存储最终生成的上下文和响应对。
    # 从MIN_CONTEXT_TURNS开始循环到轮次的数量。这样可以确保每个生成的对话对都有足够的上下文。
    for i in range(MIN_CONTEXT_TURNS, len(turns)):
        context_turns = turns[max(0, i - MAX_HISTORY):i] # 获取当前轮次前的若干轮（最多为 MAX_HISTORY），作为上下文。
        response_turn = turns[i] # 当前轮次被视为模型的响应。

        # 使用列表推导式将上下文中的每一轮拼接为字符串格式，格式为 "说话者: 发言内容"。
        # 使用分隔符 tokenizer.sep_token 将这些拼接起来，形成完整的上下文字符串。
        context = tokenizer.sep_token.join(
            [f"{turn['speaker']}: {turn['text']}" for turn in context_turns]
        )
        # 将生成的上下文和响应以字典形式添加到pairs列表中
        pairs.append({
            "context": context, # 为拼接后的上下文。
            "response": f"{response_turn['speaker']}: {response_turn['text']}", # 为当前轮次的响应。
            "speaker": response_turn['speaker']  # 可选：标注当前说话者
        })
    return pairs

def evaluate_response(actionability, mistake_identification, mistake_location, providing_guidance):
    """根据评分标准判断该回复的水平"""
    score = 0

    # 为每个参数赋分
    if actionability == "Yes":
        score += 2
    elif actionability == "To some extent":
        score += 1

    if mistake_identification == "Yes":
        score += 2
    elif mistake_identification == "To some extent":
        score += 1

    if mistake_location == "Yes":
        score += 2
    elif mistake_location == "To some extent":
        score += 1

    if providing_guidance == "Yes":
        score += 2
    elif providing_guidance == "To some extent":
        score += 1

# 主处理流程
def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)  # 假设输入格式为 [{"conversation_history": "..."}]

    all_pairs = [] # 用于存储所有生成的对话对
    for item in raw_data:
        # 选出不同模型中最好的回复并添加到对话历史中
        best_responce = "hello! everyOne!!"
        for text in item["tutor_responses"]:
            print(clean_text(item["tutor_responses"][text]["response"]))


        responce = item["conversation_history"] + " \n Tutor:" + best_responce
        # 将对话历史拆分为结构化的轮次
        turns = split_conversation(responce)

        # 如果对话轮次少于最小要求，则跳过该对话
        if len(turns) < MIN_CONTEXT_TURNS + 1:
            continue
        # 生成上下文-响应对
        pairs = generate_pairs(turns)
        # 将生成的对话对添加到总列表中
        all_pairs.extend(pairs)

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"生成完成！共处理 {len(raw_data)} 个对话，得到 {len(all_pairs)} 个训练对。")


if __name__ == "__main__":
    main()