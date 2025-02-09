import tkinter as tk
from tkinter import messagebox
from ttkbootstrap import Style
from tkinter import ttk

class TextEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Question Configuration")
        self.filename = "Questions\\output.txt"
        self.file = open(self.filename, "a")
        self.questions = {} 

        self.style = Style(theme='flatly')

        self.frame = ttk.Frame(root)
        self.frame.pack(side="right", padx=20, pady=20, fill="y")

        self.text_input = ttk.Entry(self.frame, width=40)
        self.text_input.grid(row=0, column=0, pady=10, sticky="w")
        ttk.Label(self.frame, text="Question").grid(row=0, column=1, padx=5)

        self.left = ttk.Entry(self.frame, width=20)
        self.left.grid(row=1, column=0, pady=10, sticky="w")
        ttk.Label(self.frame, text="Left Answer").grid(row=1, column=1, padx=5)

        self.right = ttk.Entry(self.frame, width=20)
        self.right.grid(row=2, column=0, pady=10, sticky="w")
        ttk.Label(self.frame, text="Right Answer").grid(row=2, column=1, padx=5)

        self.parent_id_input = ttk.Entry(self.frame, width=10)
        self.parent_id_input.grid(row=3, column=0, pady=10, sticky="w")
        ttk.Label(self.frame, text="Parent ID").grid(row=3, column=1, padx=5)

        self.child_position = ttk.Combobox(
            self.frame, values=["Left", "Right"], state="readonly", width=10
        )
        self.child_position.grid(row=4, column=0, pady=10, sticky="w")
        ttk.Label(self.frame, text="Child Position").grid(row=4, column=1, padx=5)

        # Buttons
        self.submit_button = ttk.Button(self.frame, text="Submit", command=self.submit_text)
        self.submit_button.grid(row=5, column=0, pady=10)

        self.clear_button = ttk.Button(self.frame, text="Clear Insertions", command=self.clear_insertion)
        self.clear_button.grid(row=6, column=0, pady=10)

        self.clear_file_button = ttk.Button(self.frame, text="Clear File", command=self.clear_file)
        self.clear_file_button.grid(row=7, column=0, pady=10)

        self.save_exit_button = ttk.Button(self.frame, text="Save and Exit", command=self.save_and_exit)
        self.save_exit_button.grid(row=8, column=0, pady=10)

        self.exit_button = ttk.Button(self.frame, text="Exit", command=self.exit_without_save)
        self.exit_button.grid(row=9, column=0, pady=10)

        self.question_listbox = tk.Listbox(self.frame, height=15, width=50)
        self.question_listbox.grid(row=0, column=2, rowspan=10, padx=20, pady=10)

    def calculate_child_id(self, parent_id, position):
        """Calculate the ID of the child based on parent ID and position."""
        parent_id = int(parent_id)
        return 2 * parent_id if position == "Left" else 2 * parent_id + 1

    def submit_text(self):
        text = self.text_input.get().strip()
        left = self.left.get().strip()
        right = self.right.get().strip()
        parent_id = self.parent_id_input.get().strip()
        position = self.child_position.get()

        if not text or not left or not right:
            messagebox.showwarning("Warning", "Please fill all question and answer fields.")
            return

        if parent_id and not position:
            messagebox.showwarning("Warning", "Please select child position (Left or Right) when assigning a parent.")
            return

        if not parent_id: 
            question_id = 1
        else:
            try:
                question_id = self.calculate_child_id(parent_id, position)
            except ValueError:
                messagebox.showerror("Error", "Invalid Parent ID format.")
                return

        if question_id in self.questions:
            messagebox.showerror("Error", "Question ID already exists. Please choose a unique parent/position.")
            return

        self.questions[question_id] = (text, left, right)

        question_data = f"ID: {question_id} | Q: {text} | Left: {left} | Right: {right}"
        self.question_listbox.insert(tk.END, question_data)

        self.text_input.delete(0, tk.END)
        self.left.delete(0, tk.END)
        self.right.delete(0, tk.END)
        self.parent_id_input.delete(0, tk.END)
        self.child_position.set("")

        messagebox.showinfo("Success", "Question added successfully!")

    def save_and_exit(self):
        for question_id, (question, left_answer, right_answer) in self.questions.items():
            data = f"{question_id}|{question}|{left_answer}|{right_answer}\n"
            self.file.write(data)
        self.file.close()
        self.root.destroy()

    def exit_without_save(self):
        self.file.close()
        self.root.destroy()

    def clear_file(self):
        self.file.seek(0)
        self.file.truncate()
        messagebox.showinfo("Info", "File cleared successfully.")

    def clear_insertion(self):
        self.questions.clear()
        self.question_listbox.delete(0, tk.END)
        messagebox.showinfo("Info", "All inserted data cleared!")

root = tk.Tk()
app = TextEditorApp(root)
root.mainloop()
