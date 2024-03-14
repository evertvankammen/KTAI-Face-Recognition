import tkinter as tk

options = [
    "Chandler Bing",
    "Joey Tribbiani",
    "Monica Geller",
    "Phoebe Buffay",
    "Rachel Green",
    "Ross Geller",
    "Other"
]

root = None
name_entry = None
is_true = None
clicked = None


def get_input():
    global name_entry, is_true
    is_true = False
    name_entry = clicked.get()
    root.destroy()


def yes_answer():
    global is_true
    is_true = True
    root.destroy()


def start_gui(text):
    global root, clicked
    root = tk.Tk()
    root.geometry("320x240+800+300")
    label1 = tk.Label(text="Is dit " + text + " ?")
    label1.config(font=('Helvetica bold', 16))
    label1.pack()

    button = tk.Button(root, text="Ja", command=yes_answer)
    button.config(font=('Helvetica bold', 16))
    button.pack()

    clicked = tk.StringVar()
    clicked.set(options[0])
    drop = tk.OptionMenu(root, clicked, *options)
    drop.pack()
    button2 = tk.Button(root, text="Sla keuze op", command=get_input)
    button2.config(font=('Helvetica bold', 16))
    button2.pack()
    root.mainloop()
