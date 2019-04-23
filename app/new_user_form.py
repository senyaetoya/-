#!/usr/bin/python
# -*- coding: utf-8 -*-
import ctypes
import os
import sys
from tkinter import BOTH, Tk, RAISED, RIGHT, LEFT, TOP
from tkinter.ttk import Frame, Style, Label, Button, Notebook
from PIL import Image, ImageTk as itk

cwd = os.getcwd()


class CreateWindow(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.style = Style()
        self.style.theme_use("default")
        self.pack(fill=BOTH, expand=True)
        self.UI()

    def UI(self):
        if sys.platform == 'win32':
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        frame_top = Frame(self, relief=RAISED)
        notebook = Notebook(frame_top)
        self.tab1(notebook)
        self.tab2(notebook)
        notebook.pack(fill=BOTH, expand=True)
        frame_top.pack(fill=BOTH, expand=True)

    def tab1(self, notebook):
        tab1 = Frame()
        notebook.add(tab1)
        notebook.tab(0, text=' Базовая стратегия ')

        left_frame = Frame(tab1)
        formula_p1 = Label(left_frame)

        image = Image.open(os.path.join(cwd, "img/1.png"))
        factor = 0.5
        width, height = map(lambda x: int(x * factor), image.size)
        image_sized = itk.PhotoImage(image.resize((width, height), Image.ANTIALIAS))
        formula_p1.image = image_sized
        formula_p1.pack()
        left_frame.pack(side=LEFT)


        right_frame = Frame(tab1)
        box = Label(left_frame, width=40, text='sosi kekos')
        box.pack()
        right_frame.pack(side=RIGHT)

    def tab2(self, notebook):
        tab2 = Frame()
        notebook.add(tab2)
        notebook.tab(1, text=' Кредитная стратегия ')



def main():
    root = Tk()
    app = CreateWindow(root)

    position_right = int(root.winfo_screenwidth() / 2)
    position_down = int(root.winfo_screenheight() / 2)
    root.geometry("600x500+{}+{}".format(position_right, position_down))
    root.resizable(width=False, height=False)
    root.title("Поиск оптимального портфеля закупок")
    root.iconbitmap('img/icon.ico')
    root.mainloop()


if __name__ == '__main__':
    main()
