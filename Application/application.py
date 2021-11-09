import tkinter as tk
import os

root = tk.Tk()


def plot():
    os.system('python plot.py')


def model_predict():
    os.system('python model_prediction.py')


def excel():
    os.system('python post_processing.py')


canvas = tk.Canvas(root, height=100, width=450)
canvas.pack()

prediction = tk.Button(root, text="Prediction for Signal", padx=2, pady=10,
                       relief="raised", width=15, command=model_predict)
prediction.place(relx=0.1, rely=0.1)

plot = tk.Button(root, text="Plot", padx=2, pady=10, relief="raised", width=15, command=plot)
plot.place(relx=0.4, rely=0.1)

to_excel = tk.Button(root, text="Write to Excel", padx=2, pady=10, relief="raised", width=15, command=excel)
to_excel.place(relx=0.7, rely=0.1)

root.mainloop()
