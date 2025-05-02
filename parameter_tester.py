import json
import time
from itertools import product
from transformers import GenerationConfig
import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu


# 示例测试用例（数学题场景）
test_cases = [
    {
        "id": "math_01",
        "input": "解方程: 2x + 5 = 15，求x的值。",
        "expected_answer": "5"
    },
    {
        "id": "math_02",
        "input": "计算圆的面积，半径为3cm。",
        "expected_answer": "28.27"  # πr²
    },
    {
        "id": "creative_01",
        "input": "用苏格拉底式提问解释勾股定理",
        "reference": "首先，你知道直角三角形的特点吗..."  # 参考答案
    }
]

# 参数搜索空间
param_grid = {
    "temperature": [0.3, 0.7, 1.0],
    "top_p": [0.7, 0.9, 1.0],
    "top_k": [20, 50, 100],
    "do_sample": [True, False],
    "repetition_penalty": [1.0, 1.2, 1.5],
    "max_new_tokens": [128, 256]
}


class ParameterTester:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def test_parameters(self, test_cases, param_grid):
        """主测试方法"""
        results = []
        for case_id, test_case in enumerate(test_cases):
            print(f"\nTesting Case {case_id + 1}: {test_case['input'][:50]}...")

            # 生成所有参数组合
            param_combinations = self._generate_param_combinations(param_grid)

            for params in param_combinations:
                result = self._evaluate_single_case(test_case, params)
                results.append(result)

                # 实时打印当前最佳
                self._print_current_best(results)

        return self._analyze_results(results)

    def _generate_param_combinations(self, param_grid):
        """生成参数组合的笛卡尔积"""
        keys = param_grid.keys()
        values = param_grid.values()
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def _evaluate_single_case(self, test_case, params):
        """评估单个测试用例"""
        try:
            # 生成输出
            output = self._generate_text(test_case["input"], params)

            # 计算评估指标
            metrics = {
                "time": self._measure_generation_time(test_case["input"], params),
                "length": len(output),
                "repetition": self._calculate_repetition(output),
                "bleu": self._calculate_bleu(output, test_case.get("reference", "")),
                "correctness": self._check_correctness(output, test_case)
            }

            return {
                "params": params,
                "output": output,
                "metrics": metrics,
                "case_id": test_case["id"]
            }

        except Exception as e:
            print(f"Error with params {params}: {str(e)}")
            return None

    def _generate_text(self, input_text, params):
        """执行文本生成"""
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(**params)
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 以下是评估指标函数（可根据需求扩展）
    def _measure_generation_time(self, input_text, params):
        start = time.time()
        self._generate_text(input_text, params)
        return time.time() - start

    def _calculate_repetition(self, text, ngram=3):
        words = text.split()
        ngrams = [tuple(words[i:i + ngram]) for i in range(len(words) - ngram + 1)]
        return len(ngrams) - len(set(ngrams))  # 重复ngram数量

    def _calculate_bleu(self, candidate, reference):
        return sentence_bleu([reference.split()], candidate.split())

    def _check_correctness(self, output, test_case):
        """数学题专用：检查答案是否正确"""
        if "expected_answer" in test_case:
            return output.strip().endswith(test_case["expected_answer"])
        return None


def analyze_results(self, results):
    """分析测试结果"""
    # 按参数分组
    param_stats = {}
    for r in results:
        if r is None: continue

        param_key = json.dumps(r["params"], sort_keys=True)
        if param_key not in param_stats:
            param_stats[param_key] = {
                "params": r["params"],
                "count": 0,
                "avg_time": 0,
                "avg_bleu": 0,
                "correct_rate": 0
            }

        stats = param_stats[param_key]
        stats["count"] += 1
        stats["avg_time"] += r["metrics"]["time"]

        if r["metrics"]["bleu"] > 0:
            stats["avg_bleu"] += r["metrics"]["bleu"]

        if r["metrics"]["correctness"] is not None:
            stats["correct_rate"] += int(r["metrics"]["correctness"])

    # 计算平均值
    for stats in param_stats.values():
        stats["avg_time"] /= stats["count"]
        stats["avg_bleu"] /= stats["count"]
        stats["correct_rate"] /= stats["count"]

    # 转换为列表并排序
    sorted_results = sorted(
        param_stats.values(),
        key=lambda x: (-x["correct_rate"], -x["avg_bleu"], x["avg_time"])
    )

    # 可视化前10名
    self._plot_top_results(sorted_results[:10])

    return sorted_results


def _plot_top_results(self, top_results):
    """用matplotlib绘制结果"""
    params_str = [str(r["params"])[:50] + "..." for r in top_results]
    correctness = [r["correct_rate"] * 100 for r in top_results]
    bleu_scores = [r["avg_bleu"] * 100 for r in top_results]
    times = [r["avg_time"] for r in top_results]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 双Y轴图表
    ax1.set_xlabel('Parameter Combinations')
    ax1.set_ylabel('Score (%)')
    ax1.bar(params_str, correctness, color='b', alpha=0.6, label='Correctness')
    ax1.bar(params_str, bleu_scores, color='g', alpha=0.6, label='BLEU')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Time (s)')
    ax2.plot(params_str, times, 'r-', marker='o', label='Time')
    ax2.legend(loc='upper right')

    plt.title('Top Parameter Combinations Performance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()