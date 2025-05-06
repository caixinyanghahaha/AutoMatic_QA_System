import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
import threading


class ResponseGeneratorGUI:
    def __init__(self, master, generator=None):
        self.master = master # 应用程序的主窗口（根窗口）
        self.history = []
        self.setup_ui()
        self.generating = False  # 用于防止重复点击
        self.generator = generator
        self.CHECK = 0

    def setup_ui(self):
        """设置界面UI"""
        self.master.title("数学辅导对话系统")
        self.master.geometry("800x600")

        # 对话历史显示区域
        self.history_text = scrolledtext.ScrolledText( # 创建一个文本区域，用于显示对话历史。
            self.master,
            wrap=tk.WORD, # 在单词边界进行换行。
            state='disabled', # 防止用户编辑，直到启用。
            font=("Microsoft YaHei", 12)
        )
        self.history_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True) # 将小部件放置在窗口中，并设置填充和扩展选项。

        # 输入区域
        input_frame = tk.Frame(self.master)
        input_frame.pack(padx=10, pady=10, fill=tk.X)
        self.input_entry = tk.Entry( # 创建一个单行文本输入框用于用户输入。
            input_frame,
            font=("Microsoft YaHei", 12),
            width=60
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", self.on_send) # 将 Enter 键绑定到 on_send 方法。

        # 创建一个按钮，发送用户输入。
        self.send_btn = tk.Button(
            input_frame,
            text="发送",
            command=self.on_send,
            font=("Microsoft YaHei", 12)
        )
        self.send_btn.pack(side=tk.RIGHT, padx=5)

        # 状态栏
        self.status_var = tk.StringVar() # 一个变量类，保存状态栏的字符串值。
        status_bar = tk.Label( # 显示应用程序的当前状态。
            self.master,
            textvariable=self.status_var,
            relief=tk.SUNKEN, # 给标签添加 3D 效果。
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化显示
        self.status_var.set(f"状态: 准备就绪")
        welcome_msg = "=== 数学辅导对话系统 ===\n输入您的问题或想法（清空输入或关闭窗口退出）\n"
        self.append_to_history("system", welcome_msg)

    def append_to_history(self, role, content):
        """将当前对话内容输入到历史记录"""
        self.history_text.configure(state='normal') # 修改文本区域的状态以允许编辑
        prefix = "👤 用户: " if role == "user" else "🤖 助手: "
        full_text = prefix + content + "\n\n"

        self.history_text.insert(tk.END, full_text) # 向文本区域添加新行
        self.history_text.see(tk.END)  # 自动滚动到底部
        self.history_text.configure(state='disabled') # 修改文本区域的状态禁用编辑

    def on_send(self, event=None):
        """处理发送消息"""
        if self.generating: # 如果正在处理则停止发送
            return
        user_input = self.input_entry.get().strip() # 获取并修剪用户输入。如果为空则退出。
        if not user_input:
            return
        self.input_entry.delete(0, tk.END) # 清空输入框
        self.append_to_history("user", user_input) # 添加用户消息
        self.history.append({"role": "user", "content": user_input}) # 将对话插入到历史中

        # 启动生成线程
        self.status_var.set(f"状态: 正在思考...")
        self.generating = True
        threading.Thread(target=self.generate_response, daemon=True).start() # 启动新线程生成响应

    def generate_response(self):
        """生成回复的线程方法"""
        try:
            response = self.generator.generate(self.history)
            # 在主线程更新UI
            self.append_to_history("assistant", response)
            self.history.append({"role": "assistant", "content": response})
        except Exception as e:
            self.master.after(0, lambda: self.show_error(str(e))) # 使用 after 方法在主线程中调用 show_error 方法
        finally:
            self.master.after(0, lambda: self.status_var.set(f"状态: 准备就绪"))
            self.generating = False

    def show_error(self, error_msg):
        """显示错误信息"""
        messagebox.showerror("发生错误", f"生成回复时发生错误:\n{error_msg}")
        self.history = []  # 重置对话历史
        self.history_text.configure(state='normal')
        self.history_text.delete(1.0, tk.END) # 清空文本区域
        self.history_text.configure(state='disabled')

        self.status_var.set(f"状态: 准备就绪")
        self.append_to_history("system", "=== 数学辅导对话系统 ===\n输入您的问题或想法（清空输入或关闭窗口退出）\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResponseGeneratorGUI(root)
    root.mainloop()