import json
import re
from typing import List, Dict

"""
此代码用于将数据集处理为deepSeek使用的对话对。
格式为：
[
    {
        "messages": [
          {
            "role": "Tutor",
            "content": "Hi, could ......"
          },
          {
            "role": "Student",
            "content": "To tell......"
          }
        ]
    },....
]
"""

# 配置部分
INPUT_FILE = "data/mrbench_v3_devset.json"  # 输入文件路径
OUTPUT_FILE = "data/dialog_deepseek.json"  # 输出文件路径

def clean_text(text: str) -> str:
    """清洗文本，删除多余的空格和特殊字符，保持文本整洁。"""
    text = re.sub(r'\s+', ' ', text)  # 合并多余空格
    text = re.sub(r'(\u00A0|\u2028|\r\n)', ' ', text)  # 替换特殊空白符
    return text.strip()

def split_conversation(history: str) -> List[Dict]:
    """将原始对话字符串拆分为结构化轮次"""
    # 用于存储每一轮对话的信息，最终作为返回值。
    turns = [
        {
            "role": "system",
            "content": 'You are a mathematics tutoring assistant. Your role is to guide students through Socratic questioning.'
        }
    ]
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
                # 根据角色设置名称
                if current_speaker == "Tutor":
                    current_speaker = "assistant"
                elif current_speaker == "Student":
                    current_speaker = "user"

                turns.append({
                    "role": current_speaker,
                    "content": ' '.join(current_text)
                })
            current_speaker = speaker_match.group(1)
            current_text = [speaker_match.group(2)]
        else:
            current_text.append(line)

    # 在循环结束后，检查是否还有未保存的最后一轮对话，如果有，将其添加到 turns 列表中。
    if current_speaker and current_text:
        # 根据角色设置名称
        if current_speaker == "Tutor":
            current_speaker = "assistant"
        elif current_speaker == "Student":
            current_speaker = "user"

        turns.append({
            "role": current_speaker,
            "content": ' '.join(current_text)
        })
    return turns

def evaluate_response(item):
    """根据评分标准判断该回复的水平"""
    score = 0
    ML = item["Mistake_Location"]
    MI = item["Mistake_Identification"]
    PG = item["Providing_Guidance"]
    AC = item["Actionability"]

    # 为每个参数赋分
    if AC == "Yes":
        score += 2
    elif AC == "To some extent":
        score += 1
    elif AC == "No":
        score += 0

    if MI == "Yes":
        score += 2
    elif MI == "To some extent":
        score += 1
    elif MI == "No":
        score += 0

    if ML == "Yes":
        score += 2
    elif ML == "To some extent":
        score += 1
    elif ML == "No":
        score += 0

    if PG == "Yes":
        score += 2
    elif PG == "To some extent":
        score += 1
    elif PG == "No":
        score += 0

    return score

# 主处理流程
def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)  # 假设输入格式为 [{"conversation_history": "..."}]
    all_pairs = [] # 用于存储所有生成的对话对
    for item in raw_data:
        # 选出不同模型中最好的回复并添加到对话历史中
        best_responce = "hello! everyOne!!"
        score = -1
        for text in item["tutor_responses"]:
            responce_score = evaluate_response(item["tutor_responses"][text]["annotation"])
            if responce_score > score:
                score = responce_score
                best_responce = item["tutor_responses"][text]["response"]

        responce = item["conversation_history"] + " \n Tutor:" + best_responce
        # 将对话历史拆分为结构化的轮次

        text = {
            "messages": split_conversation(responce)
        }
        all_pairs.append(text)

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"生成完成！共处理 {len(raw_data)} 个对话，得到 {len(all_pairs)} 个训练对。")


if __name__ == "__main__":
    main()