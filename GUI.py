from tkinter import *     

def gui(txt="--Prediction result--"):
    root = Tk()
    root.title('Project Prediction')
    root.geometry('500x600+300+100')
    
    display_text = StringVar()
    E_A = StringVar()
    E_T = StringVar()
    G_A = StringVar()
    G_T = StringVar()
    nu_A = StringVar()
    E_11_E_33 = StringVar()
    E_22 = StringVar()
    G_13 = StringVar()
    G_21_G_23 = StringVar()
    nu_13 = StringVar()
    nu_21_nu_23 = StringVar()
    
    display_text.set(txt)
    
    def com():
        global prop_vals
        E_A = float(text1.get())
        E_T = float(text2.get())
        G_A = float(text3.get())
        G_T = float(text4.get())
        nu_A = float(text5.get())
        E_11_E_33 = float(text6.get())
        E_22 = float(text7.get())
        G_13 = float(text8.get())
        G_21_G_23 = float(text9.get())
        nu_13 = float(text10.get())
        nu_21_nu_23 = float(text11.get())
        
        prop_vals = [E_A,E_T,G_A,G_T, nu_A, E_11_E_33, E_22, G_13, G_21_G_23, nu_13, nu_21_nu_23]
        root.destroy()
        
    labl1=Label(root, text='Prediction of J_integral', font=30)
    labl1.pack()
    
    
    lbl1 = Label(root ,text="E_A: ")
    lbl1.pack()
    text1 = Entry(root, textvariable=E_A)
    text1.pack()
    
    lbl2 = Label(root,text="E_T: ")
    lbl2.pack()
    text2 = Entry(root, textvariable=E_T)
    text2.pack()
    
    lbl3 = Label(root,text="G_A: ")
    lbl3.pack()
    text3 = Entry(root, textvariable=G_A)
    text3.pack()
    
    lbl4 = Label(root,text="G_T: ")
    lbl4.pack()
    text4 = Entry(root, textvariable=G_T)
    text4.pack()
    
    lbl5 = Label(root,text="nu_A: ")
    lbl5.pack()
    text5 = Entry(root, textvariable=nu_A)
    text5.pack()
    
    lbl6 = Label(root,text="E_11_E_33: ")
    lbl6.pack()
    text6 = Entry(root, textvariable=E_11_E_33)
    text6.pack()
    
    lbl7 = Label(root,text="E_22: ")
    lbl7.pack()
    text7 = Entry(root, textvariable=E_22)
    text7.pack()
    
    lbl8 = Label(root,text="G_13: ")
    lbl8.pack()
    text8 = Entry(root, textvariable=G_13)
    text8.pack()
    
    lbl9 = Label(root,text="G_21_G_23: ")
    lbl9.pack()
    text9 = Entry(root, textvariable=G_21_G_23)
    text9.pack()
    
    lbl10 = Label(root,text="nu_13: ")
    lbl10.pack()
    text10 = Entry(root, textvariable=nu_13)
    text10.pack()
    
    lbl11 = Label(root,text="nu_21_nu_23 ")
    lbl11.pack()
    text11 = Entry(root, textvariable=nu_21_nu_23)
    text11.pack()
    
    button1 = Button(text='Press to predict', command= com)
    button1.pack()
    
    
    txt = Label(root, textvariable=display_text)
    txt.pack(pady=10)
    
    root.mainloop()
    return prop_vals

def quit_win():
    root.destroy()
    
