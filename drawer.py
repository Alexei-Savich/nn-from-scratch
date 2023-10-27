import pickle
import tkinter as tk
from tkinter import Canvas
import numpy as np
import NN

BLOCK_SIZE = 500 // 28

NEURAL_NETWORK: NN.NeuralNetwork = None
with open('mnist_86_percent', 'rb') as file:
    NEURAL_NETWORK = pickle.load(file)


def predict_probs(image_1d):
    probs = NEURAL_NETWORK.feed_forward(image_1d)
    return probs.tolist()[0]


class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("Drawing Canvas")

        self.canvas_width = BLOCK_SIZE * 28
        self.canvas_height = BLOCK_SIZE * 28

        self.canvas = Canvas(master, width=self.canvas_width + 100, height=self.canvas_height, bg="black")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        master.bind("<Return>", self.process_and_calculate)
        master.bind("<Delete>", self.clear_canvas)

        self.image = np.zeros((28, 28), dtype=np.uint8)

    def paint(self, event):
        x, y = event.x, event.y

        block_x = x // BLOCK_SIZE
        block_y = y // BLOCK_SIZE

        for dx in [0, 1]:
            for dy in [0, 1]:
                if 0 <= block_x+dx < 28 and 0 <= block_y+dy < 28:
                    self.image[block_y+dy, block_x+dx] = 255
                    self.canvas.create_rectangle((block_x+dx)*BLOCK_SIZE, (block_y+dy)*BLOCK_SIZE,
                                                 (block_x+dx+1)*BLOCK_SIZE, (block_y+dy+1)*BLOCK_SIZE,
                                                 fill="white", outline="white")

    def process_and_calculate(self, event):
        flattened_image = self.image.flatten().reshape(1, 784)

        probs = predict_probs(flattened_image)

        for i, prob in enumerate(probs):
            self.canvas.create_text(self.canvas_width + 20, i * self.canvas_height / 10 + BLOCK_SIZE // 2,
                                    text=f"{i}: {prob:.2f}", fill="white", anchor="w")

    def clear_canvas(self, event):
        self.canvas.delete("all")
        self.image.fill(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
