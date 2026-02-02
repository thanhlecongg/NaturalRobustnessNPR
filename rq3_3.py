import json
import os
from utils import evaluate_all
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
IS_OVERWIRTE = False

def visualize(result, natural_result, figname="test.png"):
    # Extracting values for visualization
    labels = ['Positive', 'Negative', 'Combined']
    all_results = [result["positive"], result["negative"], result["n_change"]]
    natural_results = [natural_result["positive"], natural_result["negative"], natural_result["n_change"]]

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create bars with custom colors and styles
    bars1 = ax.bar(x - width/2, all_results, width, label='All Result', color='#4CAF50', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, natural_results, width, label='Natural Result', color='#2196F3', edgecolor='black', linewidth=1.2)

    # Add labels and title
    ax.set_ylabel('Prediction Changes', fontsize=18)
    # ax.set_title('Comparison of Changes in All Result and Natural Result', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.legend(fontsize=18)
    ax.set_ylim(0, result["n_change"] + 10)
    ax.legend(loc='upper left', fontsize= 18, bbox_to_anchor=(0, 1))

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
    plt.savefig(figname)
    plt.clf()

print("----------------------------------------------------------------------")
print("==> [RQ3.3 - Section 5.2.3] Performance Changes")
if os.path.exists("data/results/rq4_all_results.json") and not IS_OVERWIRTE:
    with open("data/results/rq4_all_results.json", "r") as f:
        natural_results = json.load(f)
else:
    natural_results = {
        "alpharepair": evaluate_all("alpharepair", used_natural=True),
        "recoder": evaluate_all("recoder", used_natural=True),
        "sequencer": evaluate_all("sequencer", used_natural=True),
        "selfapr": evaluate_all("selfapr", used_natural=True),
        "rewardrepair": evaluate_all("rewardrepair", used_natural=True),
        "incoder": evaluate_all("incoder", used_natural=True),
        "repairllama": evaluate_all("repairllama", used_natural=True)
    }
    with open("data/results/rq4_all_results.json", "w") as f:
        json.dump(natural_results, f)
        
        
if os.path.exists("data/results/rq3_all_results.json") and not IS_OVERWIRTE:
    with open("data/results/rq3_all_results.json", "r") as f:
        all_results = json.load(f)
else:
    all_results = {
        "alpharepair": evaluate_all("alpharepair", used_natural=False),
        "recoder": evaluate_all("recoder", used_natural=False),
        "sequencer": evaluate_all("sequencer", used_natural=False),
        "selfapr": evaluate_all("selfapr", used_natural=False),
        "rewardrepair": evaluate_all("rewardrepair", used_natural=False),
        "incoder": evaluate_all("incoder", used_natural=False),
        "repairllama": evaluate_all("repairllama", used_natural=False)
    }
    with open("data/results/rq3_all_results.json", "w") as f:
        json.dump(all_results, f)
    
# Create headers
headers = ["Repair Tool", "All Plausible Rate", "Natural Plausible Rate", "All Correct Rate", "Natural Correct Rate"]

natural_plausible_drops = []
natural_correct_drops = []
plausible_drops = []
correct_drops = []
tool_names = []

# Create rows
rows = []
for tool, (result, _, _) in all_results.items():
    natural_result = natural_results[tool][0]
    plausible_rate_drop = (result["plausible_rate"] - result["original_plausible_rate"])/result["plausible_rate"]
    correct_rate_drop = (result["correct_rate"] - result["original_correct_rate"])/result["correct_rate"]
    natural_plausible_drop = (natural_result["plausible_rate"] - natural_result["original_plausible_rate"])/natural_result["plausible_rate"]
    natural_correct_drop = (natural_result["correct_rate"] - natural_result["original_correct_rate"])/natural_result["correct_rate"]
    plausible_drops.append(plausible_rate_drop * 100)
    correct_drops.append(correct_rate_drop* 100)
    natural_plausible_drops.append(natural_plausible_drop* 100)
    natural_correct_drops.append(natural_correct_drop* 100)
    row = [tool, round(-plausible_rate_drop * 100, 1), round(-natural_plausible_drop * 100, 1), round(-correct_rate_drop * 100, 1), round(-natural_correct_drop * 100, 1)]
    rows.append(row)
    tool_names.append(tool)

# init natural_improvement_PR as 2-d array with shape (len(natural_plausible_drops), len(natural_plausible_drops))
natural_improvement_PR  = np.zeros((len(natural_plausible_drops), len(natural_plausible_drops)))   
natural_improvement_CR  = np.zeros((len(natural_plausible_drops), len(natural_plausible_drops)))
improvement_PR  = np.zeros((len(plausible_drops), len(plausible_drops)))
improvement_CR  = np.zeros((len(correct_drops), len(correct_drops)))

for i in range(len(natural_plausible_drops)):
    for j in range(len(natural_plausible_drops)):
        natural_improvement_PR[i][j] = (natural_plausible_drops[j] - natural_plausible_drops[i])
        natural_improvement_CR[i][j] = (natural_correct_drops[j] - natural_correct_drops[i])
        improvement_PR[i][j] = (plausible_drops[j] - plausible_drops[i])
        improvement_CR[i][j] = (correct_drops[j] - correct_drops[i])
        
diff_PR = np.abs(improvement_PR - natural_improvement_PR)
diff_CR = np.abs(improvement_CR - natural_improvement_CR)

plt.figure(figsize=(14, 7))

# First subplot
plt.subplot(1, 2, 1)
plt.imshow(diff_PR, vmin=0, vmax=15, cmap='rainbow', interpolation='nearest')
plt.title("Differences of Relative Performance Changes", fontsize=16)
plt.colorbar().ax.tick_params(labelsize=12)
plt.xticks(range(len(natural_plausible_drops)), tool_names, rotation=45, ha='right',  fontsize=13)  # Rotate labels
plt.yticks(range(len(natural_plausible_drops)), tool_names, fontsize=13)
plt.xlabel("Repair Techniques", fontsize=13)
plt.ylabel("Repair Techniques", fontsize=13)
plt.tight_layout()

# Second subplot
plt.subplot(1, 2, 2)
plt.imshow(diff_CR, vmin=0, vmax=15, cmap='rainbow', interpolation='nearest')
plt.title("Differences of Relative Performance Changes", fontsize=14)
plt.colorbar().ax.tick_params(labelsize=12)
plt.xticks(range(len(natural_plausible_drops)), tool_names, rotation=45, ha='right', fontsize=13)  # Rotate labels
plt.yticks(range(len(natural_plausible_drops)), tool_names, fontsize=13)
plt.xlabel("Repair Techniques", fontsize=13)
plt.ylabel("Repair Techniques", fontsize=13)
plt.tight_layout()

print("Drawing Figure 6: Absolute Differences of Performance Changes ...")
plt.savefig("data/figures/fig6_absolute_differences_performance_changes.png", dpi=300)
print("=> Please see results in data/figures/fig6_absolute_differences_performance_changes.png")

# Generate table
table = tabulate(rows, headers, tablefmt="grid")

print("Table 16: Performance Changes on Plausible and Correct Rates")
print(table)