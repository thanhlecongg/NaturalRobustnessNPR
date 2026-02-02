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

def visualize_ratio(result, natural_result, figname="test.png"):
    # Extracting values for visualization
    labels = ['Positive', 'Negative', 'Combined']
    all_results = [result["positive_rate"], result["negative_rate"], result["change_rate"]]
    natural_results = [natural_result["positive_rate"], natural_result["negative_rate"], natural_result["change_rate"]]

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create bars with custom colors and styles
    bars1 = ax.bar(x - width/2, all_results, width, label='All Result', color='#4CAF50', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, natural_results, width, label='Natural Result', color='#2196F3', edgecolor='black', linewidth=1.2)

    # Add labels and title
    ax.set_ylabel('Ratio of Prediction Changes', fontsize=18)
    # ax.set_title('Comparison of Changes in All Result and Natural Result', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.legend(fontsize=18)
    ax.set_ylim(0, result["change_rate"] + 10)
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
print("==> [RQ3.2 - Section 5.2.2] Prediction Changes")
print("Drawing Figure 5:  Ratio of prediction changes")
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
headers = ["Repair Tool", "Positive Change", "Negative Change", "Unique Change", "Natural Positive Change", "Natural Negative Change", "Natural Unique Change"]

# Create rows
rows = []
for tool, (result, _, _) in all_results.items():
    print("Processing tool:", tool)
    row = [tool] 
    positive_change = result["positive"]
    negative_change = result["negative"]
    unique_change = result["n_change"]
    positive_change_rate = result["positive_rate"]
    negative_change_rate = result["negative_rate"]
    change_rate = result["change_rate"]
    natural_result = natural_results[tool][0]
    natural_positive_change = natural_result["positive"]
    natural_negative_change = natural_result["negative"]
    natural_unique_change = natural_result["n_change"]
    natural_positive_change_rate = natural_result["positive_rate"]
    natural_negative_change_rate = natural_result["negative_rate"]
    natural_change_rate = natural_result["change_rate"]
    row.append("{} ({}%)".format(positive_change, positive_change_rate))
    row.append("{} ({}%)".format(negative_change, negative_change_rate))
    row.append("{} ({}%)".format(unique_change, change_rate))
    row.append("{} ({}%)".format(natural_positive_change, natural_positive_change_rate))
    row.append("{} ({}%)".format(natural_negative_change, natural_negative_change_rate))
    row.append("{} ({}%)".format(natural_unique_change, natural_change_rate))
    rows.append(row)
    print("Drawing figures for tool:", tool)
    visualize_ratio(result, natural_result, "data/figures/fig5_{}_prediction_changes_ratio.png".format(tool))
    print("Figure saved in data/figures/fig5_{}_prediction_changes_ratio.png".format(tool))