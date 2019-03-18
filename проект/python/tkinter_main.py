from tkinter import *
from tkinter import ttk

main = Tk()
main.title("Задача линейного программирования")
main.geometry('500x500')



tabControl = ttk.Notebook(main)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tabControl.add(tab1, text='Tab1')
tabControl.add(tab2, text='Tab2')


main.mainloop()

