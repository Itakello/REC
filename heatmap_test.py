import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Generate sample data
x_labels = ["A", "B", "C", "D"]
y_labels = ["W", "X", "Y", "Z"]
data = np.random.rand(len(x_labels), len(y_labels))

# Create the heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(
    data, annot=True, fmt=".2f", xticklabels=x_labels, yticklabels=y_labels
)
plt.title("Sample Heatmap")
plt.xlabel("X Axis Label")  # Add label for the x-axis
plt.ylabel("Y Axis Label")  # Add label for the y-axis
# Show the plot
plt.savefig()
