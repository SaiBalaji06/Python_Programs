import tkinter as tk

# Create the Tkinter window
root = tk.Tk()
root.title("Grid Placement Example")

# Create labels
label1 = tk.Label(root, text="Label 1")
label2 = tk.Label(root, text="Label 2")
label3 = tk.Label(root, text="Label 3")
label4 = tk.Label(root, text="Label 4")
label5 = tk.Label(root, text="Label 5")
label6 = tk.Label(root, text="Label 6")

# Grid Placement
label1.grid(row=0, column=0)  # Placed in the first row, first column
label2.grid(row=0, column=1)  # Placed in the first row, second column
label3.grid(row=0, column=2)  # Placed in the first row, third column
label4.grid(row=1, column=0)  # Placed in the second row, first column
label5.grid(row=1, column=1)  # Placed in the second row, second column
label6.grid(row=1, column=2)  # Placed in the second row, third column

# Start the Tkinter event loop
root.mainloop()
