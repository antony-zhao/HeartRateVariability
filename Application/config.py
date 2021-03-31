import json
from tkinter import *
from tkinter.ttk import *

config_file = open("config.json", "r")
config = json.load(config_file)
# locals().update(config)
"""
T = 0.1  # Sample Period
fs = 4000.0  # sample rate, Hz
low_cutoff = 200  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
high_cutoff = 5
nyq = 0.5 * fs  # Nyquist Frequency
order = 4  # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples
"""
root = Tk()
root.geometry("400x400")

root.title("Configure parameters")


def update_interval_length():
    config["interval_length"] = spinbox.get()
    print(config["interval_length"])


w = Label(root, text="Interval Length", font="50")
w.pack()
spinbox = Spinbox(root, from_=0, to=1000, command=update_interval_length)

spinbox.insert(0, config["interval_length"])
spinbox.pack()


root.mainloop()

