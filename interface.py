from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_SAVE_PATH = "./model/education_dialog_model"

class EducationChatbot():
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def respond(self, context, max_length=100):
        # 编码输入并显式创建attention mask
        input_text = f"{context}{self.tokenizer.eos_token}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # 生成回复，使用 with torch.no_grad() 减少内存占用，及时清理中间变量
        # with torch.no_grad():
        #     outputs = self.model.generate(
        #         input_ids=inputs.input_ids,
        #         attention_mask=inputs.attention_mask,
        #         max_length=max_length,
        #         pad_token_id=self.tokenizer.eos_token_id,
        #         do_sample=True,
        #         top_k=50,
        #         top_p=0.95,
        #         temperature=0.7
        #     )
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        # 解码并清理回复
        full_response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return full_response[len(input_text):].strip()


if __name__ == "__main__":
    # 初始化时确保MPS设备可用
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    bot = EducationChatbot()
    print("教育对话系统已加载，输入'退出'结束对话")
    while True:
        try:
            user_input = input("学生: ")
            if user_input.lower() in ["退出", "exit"]:
                break
            response = bot.respond(f"学生: {user_input}")
            print(f"导师: {response}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            torch.mps.empty_cache()