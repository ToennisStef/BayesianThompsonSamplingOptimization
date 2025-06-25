import numpy as np
import matplotlib.pyplot as plt
import ternary  # pip install python-ternary

# Define the function on the simplex
def f(x):
    return 1 - np.sum(x**2)

# Set resolution
scale = 100  # Higher = finer grid
data = {}

# Generate data over the simplex
for i in range(scale + 1):
    for j in range(scale + 1 - i):
        k = scale - i - j
        x = np.array([i, j, k]) / scale
        data[(i, j)] = f(x)

# Plot using ternary
fig, tax = ternary.figure(scale=scale)
tax.heatmap(data, style="triangular", cmap="viridis")
tax.boundary(linewidth=1.5)
tax.gridlines(color="grey", multiple=10)
tax.set_title(r"$f(\mathbf{x}) = 1 - \sum x_i^2$", fontsize=14)
tax.ticks(axis='lbr', linewidth=1, multiple=20)
tax.clear_matplotlib_ticks()
plt.tight_layout()
plt.show()
