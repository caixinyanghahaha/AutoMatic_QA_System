import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
import threading


class ResponseGeneratorGUI:
    def __init__(self, master, generator):
        self.master = master
        self.history = []
        self.setup_ui()
        self.generating = False  # 用于防止重复点击
        self.generator = generator

    def setup_ui(self):
        """设置界面元素"""
        self.master.title("数学辅导对话系统")
        self.master.geometry("800x600")

        # 对话历史显示区域
        self.history_text = scrolledtext.ScrolledText(
            self.master,
            wrap=tk.WORD,
            state='disabled',
            font=("Microsoft YaHei", 12)
        )
        self.history_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 输入区域
        input_frame = tk.Frame(self.master)
        input_frame.pack(padx=10, pady=10, fill=tk.X)

        self.input_entry = tk.Entry(
            input_frame,
            font=("Microsoft YaHei", 12),
            width=60
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", self.on_send)

        self.send_btn = tk.Button(
            input_frame,
            text="发送",
            command=self.on_send,
            font=("Microsoft YaHei", 12)
        )
        self.send_btn.pack(side=tk.RIGHT, padx=5)

        # 状态栏
        self.status_var = tk.StringVar()
        status_bar = tk.Label(
            self.master,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化显示
        self.status_var.set(f"状态: {"准备就绪"}")
        welcome_msg = "=== 数学辅导对话系统 ===\n输入您的问题或想法（清空输入或关闭窗口退出）\n"
        self.append_to_history("system", welcome_msg)

    def append_to_history(self, role, content):
        self.history_text.configure(state='normal')
        prefix = "👤 用户: " if role == "user" else "🤖 助手: "
        self.history_text.insert(tk.END, prefix + content + "\n\n")
        self.history_text.see(tk.END)  # 自动滚动到底部
        self.history_text.configure(state='disabled')

    def on_send(self, event=None):
        """处理发送消息"""
        if self.generating:
            return

        user_input = self.input_entry.get().strip()
        if not user_input:
            return

        # 清空输入框
        self.input_entry.delete(0, tk.END)

        # 添加用户消息
        self.append_to_history("user", user_input)
        self.history.append({"role": "user", "content": user_input})

        # 启动生成线程
        self.status_var.set(f"状态: {"正在思考..."}")
        self.generating = True
        threading.Thread(target=self.generate_response, daemon=True).start()

    def generate_response(self):
        """生成回复的线程方法"""
        try:
            self.generator.generate(self.history)
            response = "这是模拟回复\n换行测试"  # 示例回复

            # 在主线程更新UI
            """显示助手回复"""
            self.append_to_history("assistant", response)
            self.history.append({"role": "assistant", "content": response})

        except Exception as e:
            self.master.after(0, lambda: self.show_error(str(e)))
        finally:
            self.master.after(0, lambda: self.status_var.set(f"状态: {"准备就绪"}"))
            self.generating = False

    def show_error(self, error_msg):
        """显示错误信息"""
        messagebox.showerror("发生错误", f"生成回复时发生错误:\n{error_msg}")
        self.history = []  # 重置对话历史
        self.history_text.configure(state='normal')
        self.history_text.delete(1.0, tk.END)
        self.history_text.configure(state='disabled')
        self.show_welcome()


if __name__ == "__main__":
    root = tk.Tk()
    app = ResponseGeneratorGUI(root)
    root.mainloop()