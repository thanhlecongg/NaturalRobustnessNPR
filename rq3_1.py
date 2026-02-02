import numpy as np
from matplotlib import pyplot as plt

print("----------------------------------------------------------------------")
print("==> [RQ3.1 - Section 5.2.1] Applicability")
print("Drawing Figure 3: Bug Distribution")

# Extracting values for visualization
labels = ['Naming', 'Expression', 'Statement', 'All']
result1 = [209, 76, 71, 220]
result2 = [192, 61, 18, 204]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

# Create a bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Create bars with custom colors and styles
bars1 = ax.bar(x - width/2, result1, width, label='All Transformations', color='#4CAF50', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, result2, width, label='Natural Transformations', color='#2196F3', edgecolor='black', linewidth=1.2)

# Add labels and title
ax.set_ylabel('Numbers of Applicable Bugs', fontsize=16)
# ax.set_title('Comparison of Changes in All Result and Natural Result', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.legend(fontsize=16)
ax.set_ylim(0, 300)
ax.legend(loc='upper left', fontsize= 16, bbox_to_anchor=(0, 1))

# Adding value annotations on the bars
def add_value_labels(bars):
    """Add value labels on top of the bars"""
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16)

add_value_labels(bars1)
add_value_labels(bars2)

# Customize the grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("data/figures/fig3_bug_distribution.png", dpi=300)
plt.clf()
print("==> Figure 3 is saved to data/figures/fig3_bug_distribution.png")