import tkinter as tk
from tkinter import ttk, simpledialog
from time import sleep
from modelTrain3 import NeuralNetwork
import torch
import numpy as np
from pylsl.pylsl import StreamInlet, resolve_stream
import mne
from binary_tree import QuestionTree, Question
import translate as tl
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Response file and tree setup
RESPONSE_FILENAME = "Responses\\responses.txt"
tree = QuestionTree()
responses = []
questions = []
language = "en"
translated_questions = {}
model = None
inlet = None
initialization_complete = threading.Event()

def get_eeg_info():
    ch_names = ['Fp1.', 'Fp2.', 'F3..', 'F4..', 'C3..', 'C4..', 'F7..', 'F8..']
    ch_type = ['eeg'] * 8
    sfreq = 129
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_type)
    return info

def get_streams(name='type', type='EEG'):
    streams = resolve_stream(name, type)
    if len(streams) == 0:
        print("No stream detected. Closing the program. Terminating.")
        exit(1)
    return streams

def preprocess_signal(raw_signal):
    ch_names = ['Fp1.', 'Fp2.', 'F3..', 'F4..', 'C3..', 'C4..', 'F7..', 'F8..']
    sfreq = 129
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * 8)
    raw = mne.io.RawArray(raw_signal, info)
    raw.filter(l_freq=1, h_freq=40, picks="eeg")
    data = raw.get_data()
    normalized_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    return normalized_data

def get_data_from_stream(inlet: StreamInlet, chunk_size=129):
    samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=chunk_size)
    if len(samples) == 0:
        print("No samples received from connection. Terminating.")
        exit(1)
    data = np.array(samples).T
    preprocessed_data = preprocess_signal(data)
    return torch.tensor(preprocessed_data, dtype=torch.float32).unsqueeze(0)

def predict_intended_action(model: NeuralNetwork, input_signal):
    with torch.no_grad():
        output = model(input_signal)
        _, prediction = torch.max(output, dim=1)
        action = prediction.item()
    return action

def get_data_and_predict():
    global inlet, model
    input_signal = get_data_from_stream(inlet)
    action = predict_intended_action(model, input_signal=input_signal)
    return action

def load_model():
    global model, inlet
    try:
        # Load and compile the model
        model = NeuralNetwork(3, 129)
        model.load_state_dict(torch.load("models\\other_model_3.pth"))
        
        try:
            model = torch.jit.script(model)
        except Exception as e:
            print(f"Scripting failed: {e}")
            example_input = torch.rand(1, 8, 129)
            model = torch.jit.trace(model, example_input)
        
        torch.jit.save(model, "compiled_model.pth")
        model = torch.jit.load("compiled_model.pth")
        model.eval()
        
        # Initialize EEG stream
        streams = get_streams()
        inlet = StreamInlet(streams[0])
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def translate_tree_recursive(node, translator):
    if node is None:
        return {}
    
    translations = {}
    
    # Translate current node
    node_id = id(node)
    translations[f"q_{node_id}"] = translator.translate(node.question)
    translations[f"l_{node_id}"] = translator.translate(node.left_answer)
    translations[f"r_{node_id}"] = translator.translate(node.right_answer)
    
    # Recursively translate children
    if node.left:
        translations.update(translate_tree_recursive(node.left, translator))
    if node.right:
        translations.update(translate_tree_recursive(node.right, translator))
    
    return translations

def translate_tree():
    global translated_questions, language
    if language == "en":
        return {}
    
    try:
        translator = tl.Translator(to_lang=language)
        return translate_tree_recursive(tree.root, translator)
    except Exception as e:
        print(f"Translation error: {e}")
        return {}

def initialize_system():
    global translated_questions
    
    try:
        # Load model and translate tree concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            model_future = executor.submit(load_model)
            translation_future = executor.submit(translate_tree)
            
            # Wait for both tasks to complete
            model_loaded = model_future.result()
            translated_questions = translation_future.result()
            
            if not model_loaded:
                raise Exception("Model loading failed")
                
        initialization_complete.set()
        return True
    except Exception as e:
        print(f"Initialization error: {e}")
        return False

def show_instructions():
    root = tk.Tk()
    root.title("Instructions")
    root.geometry("800x600")
    root.configure(bg="#f7f7f7")

    loading_label = tk.Label(
        root,
        text="Loading system...",
        font=("Helvetica", 12),
        fg="#666",
        bg="#f7f7f7"
    )
    loading_label.pack(side="top", pady=10)

    instructions = (
        "Welcome to the Adaptive Questionnaire!\n\n\n"
        "Instructions:\n"
        "- Think about tightening your left fist to select the LEFT option.\n\n\n"
        "- Think about tightening your right fist to select the RIGHT option.\n\n\n"
        "- If no action is taken, the system will wait.\n\n"
        "The questionnaire will begin shortly."
    )

    label = tk.Label(
        root,
        text=instructions,
        font=("Helvetica", 16),
        fg="#333",
        bg="#f7f7f7",
        wraplength=600,
        justify="center",
    )
    label.pack(expand=True)

    # Start initialization in background
    threading.Thread(target=initialize_system, daemon=True).start()
    
    def check_initialization():
        if initialization_complete.is_set():
            root.after(1000, lambda: [root.destroy(), start_questionnaire()])
        else:
            root.after(100, check_initialization)
    
    root.after(100, check_initialization)
    root.mainloop()

def get_translated_text(node, text_type):
    if node is None:
        return ""
        
    node_id = id(node)
    if language == "en":
        if text_type == "question":
            return node.question
        elif text_type == "left":
            return node.left_answer
        else:
            return node.right_answer
    
    try:
        if text_type == "question":
            return translated_questions.get(f"q_{node_id}", node.question)
        elif text_type == "left":
            return translated_questions.get(f"l_{node_id}", node.left_answer)
        else:
            return translated_questions.get(f"r_{node_id}", node.right_answer)
    except:
        # Fallback to original text if translation fails
        if text_type == "question":
            return node.question
        elif text_type == "left":
            return node.left_answer
        else:
            return node.right_answer

def save_responses():
    try:
        with open(RESPONSE_FILENAME, 'w', encoding='utf-8') as file:
            for question, response in zip(questions, responses):
                file.write(f"{question}|{response}\n")
    except Exception as e:
        print(f"Error saving responses: {e}")

def start_questionnaire():
    global current_node

    root = tk.Tk()
    root.title("Adaptive Questionnaire")
    root.geometry("800x600")
    root.configure(bg="#ffffff")

    def end_questionnaire():
        save_responses()
        root.destroy()

    def update_question():
        if current_node is None:
            end_questionnaire()
            return False
        
        question_label.config(text=get_translated_text(current_node, "question"))
        left_button.config(text=get_translated_text(current_node, "left"))
        right_button.config(text=get_translated_text(current_node, "right"))
        return True

    def on_left():
        global current_node
        if current_node:
            responses.append(current_node.left_answer)
            questions.append(current_node.question)
            current_node = current_node.left
            if not update_question():
                end_questionnaire()

    def on_right():
        global current_node
        if current_node:
            responses.append(current_node.right_answer)
            questions.append(current_node.question)
            current_node = current_node.right
            if not update_question():
                end_questionnaire()

    def no_action():
        print("No action taken")

    def simulate_button_click():
        if current_node is None:
            return
        
        action = get_data_and_predict()
        if action == 1:
            on_left()
        elif action == 2:
            on_right()
        else:
            no_action()
            
        if current_node is not None:
            root.after(2000, simulate_button_click)

    header = tk.Label(
        root,
        text="Adaptive Questionnaire",
        font=("Helvetica", 24, "bold"),
        fg="#333",
        bg="#ffffff",
    )
    header.pack(pady=20)

    question_label = tk.Label(
        root,
        text=get_translated_text(current_node, "question"),
        font=("Helvetica", 18),
        fg="#333",
        bg="#ffffff",
        wraplength=700,
        justify="center",
    )
    question_label.pack(pady=30)

    button_frame = tk.Frame(root, bg="#ffffff")
    button_frame.pack(pady=50)

    left_button = tk.Button(
        button_frame,
        text=get_translated_text(current_node, "left"),
        font=("Helvetica", 16),
        bg="#d4edda",
        fg="#155724",
        activebackground="#c3e6cb",
        activeforeground="#155724",
        width=15,
        height=2,
        command=on_left,
    )
    left_button.pack(side="left", padx=20)

    right_button = tk.Button(
        button_frame,
        text=get_translated_text(current_node, "right"),
        font=("Helvetica", 16),
        bg="#f8d7da",
        fg="#721c24",
        activebackground="#f5c6cb",
        activeforeground="#721c24",
        width=15,
        height=2,
        command=on_right,
    )
    right_button.pack(side="right", padx=20)

    root.after(2000, simulate_button_click)
    root.mainloop()

def main():
    global current_node, language

    root = tk.Tk()
    root.withdraw()
    file_prompt = "Enter the name of the file to load questions from:"
    question_file = simpledialog.askstring("Input", file_prompt)

    if not question_file:
        print("No file provided. Exiting.")
        return

    language_window = tk.Toplevel()
    language_window.title("Select Language")
    language_var = tk.StringVar(value="en")

    ttk.Label(language_window, text="Select your preferred language:").pack(pady=10)
    language_dropdown = ttk.Combobox(
        language_window, 
        values=["en", "hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa", "ur", 
                "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ar"], 
        textvariable=language_var,
        state="readonly"
    )

    language_dropdown.pack(pady=10)
    ttk.Button(language_window, text="Confirm", command=language_window.destroy).pack(pady=10)
    language_window.wait_window()

    language = language_var.get()

    try:
        tree.build_tree_from_file(f"Questions\\{question_file}")
        current_node = tree.root
        show_instructions()
    except FileNotFoundError:
        print(f"File '{question_file}' not found. Exiting.")
        return

if __name__ == "__main__":
    main()