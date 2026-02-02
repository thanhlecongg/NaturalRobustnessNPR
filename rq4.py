import json
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import os
from utils import *

USED_NATURAL = False
IS_OVERWIRTE = True
N_BUGS = 204

print("==> [RQ4 - Section 5.3] Natural Robustness of Neural Program Repair Techniques")

print("----------------------------------------------------------------------")
print("Table 17: Prediction Changes on Plausible and Correct Rates")
print("----------------------------------------------------------------------")

if os.path.exists("data/results/rq4_all_results.json") and not IS_OVERWIRTE:
    with open("data/results/rq4_all_results.json", "r") as f:
        all_results = json.load(f)
else:
    all_results = {
        "alpharepair": evaluate_all("alpharepair", used_natural=True),
        "recoder": evaluate_all("recoder", used_natural=True),
        "sequencer": evaluate_all("sequencer", used_natural=True),
        "selfapr": evaluate_all("selfapr", used_natural=True),
        "rewardrepair": evaluate_all("rewardrepair", used_natural=True),
        "incoder": evaluate_all("incoder", used_natural=True),
        "repairllama": evaluate_all("repairllama", used_natural=True)
    }
    with open("data/results/rq4_all_results.json", "w") as f:
        json.dump(all_results, f)

# Create headers
headers = ["Repair Tool", "Postive", "Negative", "All"]

# Create rows
rows = []
for tool, (result, all_positive_changes, all_negative_changes) in all_results.items():
    row = [tool] 
    positive_change = result["positive"]
    negative_change = result["negative"]
    unique_change = result["n_change"]
    positive_change_rate = result["positive_rate"]
    negative_change_rate = result["negative_rate"]
    avg_positive = sum([len(all_positive_changes[bug_id]) for bug_id in all_positive_changes])/positive_change
    avg_negative = sum([len(all_negative_changes[bug_id]) for bug_id in all_negative_changes])/negative_change
    avg_all = (sum([len(all_positive_changes[bug_id]) for bug_id in all_positive_changes]) + sum([len(all_negative_changes[bug_id]) for bug_id in all_negative_changes]))/unique_change
    change_rate = result["change_rate"]
    avg_negative = round(avg_negative, 1)
    avg_positive = round(avg_positive, 1)
    avg_all = round(avg_all, 1)
    row.append("{} ({}, {})".format(positive_change, positive_change_rate, avg_positive))
    row.append("{} ({}, {})".format(negative_change, negative_change_rate, avg_negative))
    row.append("{} ({}, {})".format(unique_change, change_rate, avg_all))
    rows.append(row)

# Generate table
table = tabulate(rows, headers, tablefmt="grid")

print(table)

print("----------------------------------------------------------------------")
print("Table 18: Prediction Changes per Transformation Categories")
print("----------------------------------------------------------------------")

if os.path.exists("data/results/rq4_changes_per_categories.json") and not IS_OVERWIRTE:
    with open("data/results/rq4_changes_per_categories.json", "r") as f:
        changes_per_category = json.load(f)
else:   
    changes_per_category = {
        "alpharepair": {
            "naming": evaluate("alpharepair", "naming", used_natural=True),
            "expression": evaluate("alpharepair", "expression", used_natural=True),
            "statement": evaluate("alpharepair", "statement", used_natural=True)
        },
        "recoder": {
            "naming": evaluate("recoder", "naming", used_natural=True),
            "expression": evaluate("recoder", "expression", used_natural=True),
            "statement": evaluate("recoder", "statement", used_natural=True)
        },
        "sequencer": {
            "naming": evaluate("sequencer", "naming", used_natural=True),
            "expression": evaluate("sequencer", "expression", used_natural=True),
            "statement": evaluate("sequencer", "statement", used_natural=True)
        },
        "selfapr": {
            "naming": evaluate("selfapr", "naming", used_natural=True),
            "expression": evaluate("selfapr", "expression", used_natural=True),
            "statement": evaluate("selfapr", "statement", used_natural=True)
        },
        "rewardrepair": {
            "naming": evaluate("rewardrepair", "naming", used_natural=True),
            "expression": evaluate("rewardrepair", "expression", used_natural=True),
            "statement": evaluate("rewardrepair", "statement", used_natural=True)
        },
        "incoder": {
            "naming": evaluate("incoder", "naming", used_natural=True),
            "expression": evaluate("incoder", "expression", used_natural=True),
            "statement": evaluate("incoder", "statement", used_natural=True)
        },
        "repairllama": {
            "naming": evaluate("repairllama", "naming", used_natural=True),
            "expression": evaluate("repairllama", "expression", used_natural=True),
            "statement": evaluate("repairllama", "statement", used_natural=True)
        }
    }
    
    with open("data/results/rq4_changes_per_categories.json", "w") as f:
        json.dump(changes_per_category, f)

# Create headers
headers = ["Repair Tool", "Naming", "Expression", "Statement"]

# Create rows
rows = []
for tool, transformations in changes_per_category.items():
    row = [tool] 
    for transform in ["naming", "expression", "statement"]:
        result = transformations[transform]
        unique_change = result["n_change"]
        change_rate = result["change_rate"]
        row.append("{} ({})".format(unique_change, change_rate))
    
    rows.append(row)

# Generate table
table = tabulate(rows, headers, tablefmt="grid")

print(table)

print("----------------------------------------------------------------------")
print("==> Table 19: Impact on NPR Performance")
print("----------------------------------------------------------------------")

# Create headers
headers = ["Repair Tool", "Plausible Rate Before", "Plausible Rate After", "Changes", "Correct Rate Before", "Correct Rate After", "Changes"]

# Create rows
rows = []
original_plausible_rates = []
original_correct_rates = []
plausible_rates = []
correct_rates = []
tool_names = []
for tool, results in all_results.items():
    original_plausible_rate = round((results[0]["original_plausible_rate"]/N_BUGS)*100, 1)
    original_correct_rate = round((results[0]["original_correct_rate"]/N_BUGS)*100, 1)
    plausible_rate = round((results[0]["plausible_rate"]/N_BUGS)*100, 1)
    correct_rate = round((results[0]["correct_rate"]/N_BUGS)*100, 1)
    plausible_changes = (plausible_rate - original_plausible_rate)/original_plausible_rate
    correct_changes = (correct_rate - original_correct_rate)/original_correct_rate
    plausible_changes = round(plausible_changes*100, 1)
    correct_changes = round(correct_changes*100, 1) 
    row = [tool, "{}%".format(original_plausible_rate), "{}%".format(plausible_rate), "{}%".format(plausible_changes), "{}%".format(original_correct_rate), "{}%".format(correct_rate), "{}%".format(correct_changes)]
    original_plausible_rates.append(original_plausible_rate)
    original_correct_rates.append(original_correct_rate)
    plausible_rates.append(plausible_rate)
    correct_rates.append(correct_rate)
    rows.append(row)
    tool_names.append(tool)
    
# init original_improvement_PR as 2-d array with shape (len(original_plausible_rates), len(original_plausible_rates))
original_improvement_PR  = np.zeros((len(original_plausible_rates), len(original_plausible_rates)))   
original_improvement_CR  = np.zeros((len(original_plausible_rates), len(original_plausible_rates)))
improvement_PR  = np.zeros((len(plausible_rates), len(plausible_rates)))
improvement_CR  = np.zeros((len(correct_rates), len(correct_rates)))

for i in range(len(original_plausible_rates)):
    for j in range(len(original_plausible_rates)):
        original_improvement_PR[i][j] = (original_plausible_rates[j] - original_plausible_rates[i])
        original_improvement_CR[i][j] = (original_correct_rates[j] - original_correct_rates[i])
        improvement_PR[i][j] = (plausible_rates[j] - plausible_rates[i])
        improvement_CR[i][j] = (correct_rates[j] - correct_rates[i])
        
diff_PR = np.abs(improvement_PR - original_improvement_PR)
diff_CR = np.abs(improvement_CR - original_improvement_CR) 

plt.figure(figsize=(14, 7))

# First subplot
plt.subplot(1, 2, 1)
plt.imshow(diff_PR, vmin=0, vmax=5, cmap='rainbow', interpolation='nearest')
plt.title("Differences on Plausible Rate Improvement", fontsize=16)
plt.colorbar().ax.tick_params(labelsize=12)
plt.xticks(range(len(original_plausible_rates)), tool_names, rotation=45, ha='right',  fontsize=13)  # Rotate labels
plt.yticks(range(len(original_plausible_rates)), tool_names, fontsize=13)
plt.xlabel("Repair Techniques", fontsize=13)
plt.ylabel("Repair Techniques", fontsize=13)
plt.tight_layout()

# Second subplot
plt.subplot(1, 2, 2)
plt.imshow(diff_CR, vmin=0, vmax=5, cmap='rainbow', interpolation='nearest')
plt.title("Differences of Correct Rate Improvement", fontsize=14)
plt.colorbar().ax.tick_params(labelsize=12)
plt.xticks(range(len(original_plausible_rates)), tool_names, rotation=45, ha='right', fontsize=13)  # Rotate labels
plt.yticks(range(len(original_plausible_rates)), tool_names, fontsize=13)
plt.xlabel("Repair Techniques", fontsize=13)
plt.ylabel("Repair Techniques", fontsize=13)
plt.tight_layout()

plt.savefig("data/results/improvement_diffs.png", dpi=300)


# Generate table
table = tabulate(rows, headers, tablefmt="grid")
print(table)