import tkinter as tk
from tkinter import Canvas
from time import sleep
import numpy as np
import torch
from pylsl.pylsl import StreamInlet, resolve_stream
from threading import Thread, Event
from modelTrain3 import NeuralNetwork

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PLAYER_SIZE = 50
OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 50
OBSTACLE_SPEED = 5
FPS = 20

player_x = WINDOW_WIDTH // 2
player_y = WINDOW_HEIGHT - PLAYER_SIZE - 10
obstacles = []
game_over = False

inlet = None
model = None
eeg_thread_ready = Event()

def get_streams(name="type", type="EEG"):
    streams = resolve_stream(name, type)
    if len(streams) == 0:
        exit(1)
    return streams

def preprocess_signal(raw_signal):
    normalized_data = (raw_signal - np.mean(raw_signal, axis=1, keepdims=True)) / np.std(
        raw_signal, axis=1, keepdims=True
    )
    return normalized_data

def get_data_from_stream(inlet: StreamInlet, chunk_size=129):
    samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=chunk_size)
    if len(samples) == 0:
        return None
    data = np.array(samples).T
    preprocessed_data = preprocess_signal(data)
    return torch.tensor(preprocessed_data, dtype=torch.float32).unsqueeze(0)

def predict_action(model: NeuralNetwork, input_signal):
    if input_signal is None:
        return 0
    with torch.no_grad():
        output = model(input_signal)
        _, prediction = torch.max(output, dim=1)
        return prediction.item()

def initialize_model():
    global model, inlet
    try:
        model = NeuralNetwork(3, 129)
        model = torch.jit.load("compiled_model.pth")
        model.eval()
        inlet = StreamInlet(get_streams()[0])
        eeg_thread_ready.set()
    except Exception as e:
        exit(1)

def create_obstacle():
    x = np.random.randint(0, WINDOW_WIDTH - OBSTACLE_WIDTH)
    return {"x": x, "y": -OBSTACLE_HEIGHT}

def move_obstacles():
    global game_over
    for obstacle in obstacles:
        obstacle["y"] += OBSTACLE_SPEED
        if obstacle["y"] + OBSTACLE_HEIGHT > player_y:
            if (
                player_x < obstacle["x"] + OBSTACLE_WIDTH
                and player_x + PLAYER_SIZE > obstacle["x"]
            ):
                game_over = True

def draw_game(canvas):
    canvas.delete("all")
    canvas.create_rectangle(
        player_x, player_y, player_x + PLAYER_SIZE, player_y + PLAYER_SIZE, fill="blue"
    )
    for obstacle in obstacles:
        canvas.create_rectangle(
            obstacle["x"],
            obstacle["y"],
            obstacle["x"] + OBSTACLE_WIDTH,
            obstacle["y"] + OBSTACLE_HEIGHT,
            fill="red",
        )

def eeg_thread():
    global player_x, obstacles, game_over
    eeg_thread_ready.wait()
    while not game_over:
        input_signal = get_data_from_stream(inlet)
        action = predict_action(model, input_signal)
        print(action)
        if action == 1:
            player_x = max(0, player_x - 20)
        elif action == 2:
            player_x = min(WINDOW_WIDTH - PLAYER_SIZE, player_x + 20)
        sleep(0.05)

def game_loop(canvas, root):
    global obstacles, game_over
    if np.random.random() < 0.05:
        obstacles.append(create_obstacle())
    move_obstacles()
    obstacles = [obs for obs in obstacles if obs["y"] < WINDOW_HEIGHT]
    draw_game(canvas)
    if game_over:
        canvas.create_text(
            WINDOW_WIDTH // 2,
            WINDOW_HEIGHT // 2,
            text="GAME OVER",
            font=("Helvetica", 40),
            fill="black",
        )
        return
    root.after(int(1000 / FPS), game_loop, canvas, root)

def main():
    global game_over
    model_thread = Thread(target=initialize_model, daemon=True)
    model_thread.start()
    eeg_processing_thread = Thread(target=eeg_thread, daemon=True)
    eeg_processing_thread.start()
    root = tk.Tk()
    root.title("EEG-Controlled Game")
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    root.resizable(False, False)
    canvas = Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
    canvas.pack()
    root.after(int(1000 / FPS), game_loop, canvas, root)
    root.mainloop()

if __name__ == "__main__":
    main()
