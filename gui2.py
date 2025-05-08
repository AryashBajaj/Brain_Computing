import tkinter as tk
from tkinter import ttk, simpledialog
from time import sleep
from modelTrain3 import NeuralNetwork
import torch
import numpy as np
from pylsl.pylsl import StreamInlet, resolve_stream
import mne
from binary_tree import QuestionTree, Question
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from functools import lru_cache
import json
from typing import Dict, Optional
import os

RESPONSE_FILENAME = "Responses\\responses.txt"
CACHE_DIR = "TranslationCache"
tree = QuestionTree()
responses = []
questions = []
language = "en"
translated_questions = {}
model = None
inlet = None
initialization_complete = threading.Event()

os.makedirs(CACHE_DIR, exist_ok=True)

class CachedTranslator:
    def __init__(self, to_lang: str):
        self.to_lang = to_lang
        self.cache_file = os.path.join(CACHE_DIR, f"translation_cache_{to_lang}.json")
        self._cache: Dict[str, Dict[str, str]] = {}
        self._load_cache()
        
    def _load_cache(self):
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self._cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._cache = {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    @lru_cache(maxsize=1000)
    def get_cached_translation(self, text: str, to_lang: str) -> Optional[str]:
        lang_cache = self._cache.get(to_lang, {})
        return lang_cache.get(text)

    def update_cache(self, text: str, translation: str):
        if self.to_lang not in self._cache:
            self._cache[self.to_lang] = {}
        self._cache[self.to_lang][text] = translation
        
    async def translate_text(self, text: str, session: aiohttp.ClientSession) -> str:
        cached = self.get_cached_translation(text, self.to_lang)
        if cached:
            return cached
        
        try:
            async with session.post(
                'http://127.0.0.1:5000/translate',
                json={
                    'q': text,
                    'source': 'en',
                    'target': self.to_lang
                }
            ) as response:
                if response.status != 200:
                    print(f"Translation API error: Status {response.status}")
                    return text
                    
                result = await response.json()

                translation = None
                if isinstance(result, dict):
                    translation = (
                        result.get('translatedText') or  
                        result.get('translated_text') or  
                        result.get('translation') or      
                        result.get('text')               
                    )
                
                if translation:
                    self.update_cache(text, translation)
                    return translation
                else:
                    print(f"Unexpected translation response format: {result}")
                    return text
                    
        except aiohttp.ClientError as e:
            print(f"Network error during translation: {e}")
            return text
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response from translation API: {e}")
            return text
        except Exception as e:
            print(f"Unexpected error during translation: {e}")
            return text

async def translate_tree_async(tree, to_lang: str) -> Dict[str, str]:
    translator = CachedTranslator(to_lang)
    translations = {}
    
    async def process_node(node):
        if node is None:
            return
            
        node_id = id(node)
        async with aiohttp.ClientSession() as session:
            tasks = [
                translator.translate_text(node.question, session),
                translator.translate_text(node.left_answer, session),
                translator.translate_text(node.right_answer, session)
            ]
            q_trans, l_trans, r_trans = await asyncio.gather(*tasks)
            
            translations[f"q_{node_id}"] = q_trans
            translations[f"l_{node_id}"] = l_trans
            translations[f"r_{node_id}"] = r_trans
            
            if node.left or node.right:
                await asyncio.gather(
                    process_node(node.left) if node.left else asyncio.sleep(0),
                    process_node(node.right) if node.right else asyncio.sleep(0)
                )
    
    await process_node(tree.root)
    translator._save_cache()
    return translations

def translate_tree() -> Dict[str, str]:
    if language == "en":
        return {}
        
    with ThreadPoolExecutor() as executor:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(translate_tree_async(tree, language))
        finally:
            loop.close()

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
        model = NeuralNetwork(3, 129)
        model = torch.jit.load("compiled_model.pth")
        model.eval()
        
        streams = get_streams()
        inlet = StreamInlet(streams[0])
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def initialize_system():
    global translated_questions
    
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            model_future = executor.submit(load_model)
            translation_future = executor.submit(translate_tree)
            
            model_loaded = model_future.result()
            translated_questions = translation_future.result()
            
            if not model_loaded:
                raise Exception("Model loading failed")
                
        initialization_complete.set()
        return True
    except Exception as e:
        print(f"Initialization error: {e}")
        return False

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
        if text_type == "question":
            return node.question
        elif text_type == "left":
            return node.left_answer
        else:
            return node.right_answer

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
        "Instructions:\n\n\n"
        "- Think about tightening your left fist to select the LEFT option.\n\n\n"
        "- Think about tightening your right fist to select the RIGHT option.\n\n\n"
        "- If no action is taken, the system will wait.\n\n"
        "The questionnaire will begin shortly."
    )

    header = tk.Label(
        root,
        text="Welcome to the Adaptive Questionnaire!\n\n\n",
        font=("Helevetica", 24),
        fg="#333",
        bg="#f7f7f7",
        wraplength=600,
        justify="center",
    )
    header.pack(expand=True)

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

    threading.Thread(target=initialize_system, daemon=True).start()
    
    def check_initialization():
        if initialization_complete.is_set():
            root.after(3000, lambda: [root.destroy(), start_questionnaire()])
        else:
            root.after(100, check_initialization)
    
    root.after(100, check_initialization)
    root.mainloop()

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
        if root :
            save_responses()
            root.destroy()
            exit(1)


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
        font=("Helvetica", 30, "bold"),
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