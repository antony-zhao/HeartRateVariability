from matplotlib.widgets import Slider, Button, RadioButtons
import tkinter as tk

root = tk.Tk()

canvas = tk.Canvas(root, height = 540, width = 960)
canvas.pack()

train_button = tk.Button(root, text="Train Model", padx=2, pady=10, relief="raised", width=15)
train_button.place(relx=0.1, rely=0.1)

prediction = tk.Button(root, text="Prediction for Signal", padx=2, pady=10, relief="raised", width=15)
prediction.place(relx=0.25, rely=0.1)

plot = tk.Button(root, text="Plot", padx=2, pady=10, relief="raised", width=15)
plot.place(relx=0.4, rely=0.1)

root.mainloop()