# interactive_test.py

# You can run this file line-by-line in VS Code with Shift + Enter
# This behaves like Spyder's interactive console

import numpy as np
import matplotlib.pyplot as plt
# This is a comment
# You can write code here and execute# This is a comment
# You can write code here and execute
# Define a function
def square(x):
    return x ** 2

# Use the function
value = 5
squared = square(value)
print(f"{value} squared is {squared}")

# Create and display a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y, label="sin(x)")
plt.title("Interactive Plot")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.grid(True)
plt.show()
