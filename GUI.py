from tkinter import *
#from NeuroPy import NeuroPy #Neuropy is a library
#import csv
import tkinter as tk
#from PIL import ImageTk, Image
#import numpy as np
#fields = 'First Name', 'Last Name'

def fetch():
      print('%s: "%s"') 

if __name__ == '__main__':
   #root = Tk()
   root = tk.Tk()
   root.geometry("700x430")
   root.configure(background='white')
   frame = tk.Frame(root)
   frame.grid()
   frame.pack()

   Background = Frame(root, bg='white', width = 700, height= 430)
   Background.pack()
    
   Header = Frame(Background, bg='#0099FF', width = 700, height= 100)
   Header.pack(padx = 10)
    
   photo = PhotoImage(file="uni.gif")
   uni_logo = Label(Header, image=photo, bg='#0099FF')
   uni_logo.photo = photo
   uni_logo.pack(side = LEFT, padx = 0)
   
   project_name = Label(Header, bg='#0099FF', fg='#FFFFFF',text="Deep Bind Predictor", font='CenturyGothic 20 bold', width = 700)
   project_name.pack()
   project_name333 = Label(Header, bg='#0099FF', fg='#FFFFFF',text="Natinal University of Computer and Emerging Sciences", font='CenturyGothic 10', width = 700)
   project_name333.pack()

   Centre = Frame(Background, bg='white', width = 700, height=430)
   Centre.pack(pady = 30)       
   
   Footer = Frame(Centre, bg='white', width = 700, height= 100)
   Footer.pack(pady = 10)
   
   Footer1 = Frame(Centre, bg='white', width = 700, height= 100)
   Footer1.pack(pady = 10)
   
   Footer2 = Frame(Centre, bg='white', width = 700, height= 100)
   Footer2.pack(pady = 10)

   lab = Label(Footer, width=20,bg='white', text="Enter Protein id ", anchor='w')
   ent = Entry(Footer)
   lab.pack(side=TOP, fill=X, padx=5, pady=5)
   lab.pack(side=LEFT)
   ent.pack(side=RIGHT, expand=YES, fill=X )
   
   #ents = makeform(Footer, "First Name")
   #row = Frame(root)
   #root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
   b1 = Button(Footer1, bg= '#0099FF', fg = 'white' ,  text = 'Predict', font='CenturyGothic 12', command=fetch , width=10, height=1)   
   b1.pack(side  = LEFT , padx =25 , pady = 25)
   
   S = Scrollbar(Footer2)
   lab1 = Label(Footer2, width=5, bg='white',text="Result: ", anchor='w')
   lab1.pack(side=TOP, fill=X, padx=5, pady=5)
   lab1.pack(side=TOP)
   T = Text(Footer2, height=4, width=50)
   S.pack(side=RIGHT, fill=Y)
   T.pack(side=LEFT, fill=Y)
   S.config(command=T.yview)
   T.config(yscrollcommand=S.set)
   quote = """prediction result here."""
   T.insert(END, quote)
   #b2 = Button(root, text='Quit', command=root.quit)
   #b2.pack(side=LEFT, padx=5, pady=5)
   root.mainloop()