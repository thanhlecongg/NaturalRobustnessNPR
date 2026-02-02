import json
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import os
import pandas as pd
from collections import defaultdict
from scipy import stats

_ORIGINAL_MAPPING = {}

with open("data/meta/original.json", "r") as f:
    original_data = json.load(f)
mapping = {}
for bug in original_data:
    _ORIGINAL_MAPPING[bug["bug_id"]] = bug["id"]
    
valid_bugs = {}
for transform in ["naming", "expression", "statement"]:
    with open(f"data/valid_bugs/{transform}.txt", "r") as f:
        valid_bugs[transform] = []
        for line in f:
            valid_bugs[transform].append(int(line))

with open("data/valid_bugs/original.txt", "r") as f:
    valid_bugs["original"] = f.read().split("\n")


def load_naturalness_evaluation(type):
    path = "data/ids/{}_ids.txt".format(type)
    return open(path).read().split("\n")

naturalness_evaluation = {
    "unnatural": load_naturalness_evaluation("unnatural"),
    "natural": load_naturalness_evaluation("natural"),
    "likely_natural": load_naturalness_evaluation("likely_natural"),
    "likely_unnatural": load_naturalness_evaluation("likely_unnatural")
}
    
def read_meta_data(name):
    file_name = "data/meta/{}.json".format(name)
    with open(file_name, 'r') as json_file:
        json_data = json_file.read()

    # Parse the JSON data
    data = json.loads(json_data)
    
    processed_data = {}
    for bug_info in data:
        processed_data[bug_info["id"]] = bug_info
        
    
    return processed_data


def MWW(A, B):
    # Ensure both arrays are NumPy arrays
    A = np.array(A)
    B = np.array(B)

    statistic, p_value = stats.mannwhitneyu(A, B, alternative='less')
    r = 1 - 2*statistic/(A.size*B.size)


    return statistic, p_value, r

def load_naturalness_evaluation(type):
    path = "data/ids/{}_ids.txt".format(type)
    return open(path).read().split("\n")

def parse_correctness_assessment(path, tool=None):
    df = pd.read_excel(path)
    for col in df.columns:
        df = df.rename(columns={col: col.strip()})
    
    rs = {}
    diff = {}
    for idx in range(len(df["bug_id"])):
        
        if tool == "incoder" or tool == "repairllama":
            bug_id = int(df["bug_id"][idx].split("-")[1])
        elif tool == "incoder_original" or tool == "repairllama_original":
            bug_id = int(_ORIGINAL_MAPPING[df["bug_id"][idx]])
        else:
            bug_id = df["id"][idx]
        
        assert "Annotation" in df.columns
        if tool in ["incoder", "incoder_original", "repairllama", "repairllama_original"]:
            assert df["Annotation"][idx] in [0, 1]
            label = df["Annotation"][idx] == 1
        else:
            assert df["Annotation"][idx] in ["yes", "no"]
            label = df["Annotation"][idx] == "yes"
        rs[bug_id] = label
        diff[bug_id] = df["generated_diff"][idx]
    
    return rs, diff

def parse_results(transform, meta_path, annotation_path, tool, used_natural):
    meta = json.load(meta_path)
    assessment, diff = parse_correctness_assessment(annotation_path, tool)
    rs = {}
    for bug_info in meta:
        id = bug_info["id"] 
        if "{}_{}".format(transform, id) not in naturalness_evaluation["natural"] and used_natural:
            continue
        if id not in valid_bugs[transform]:
            continue

        rs[id] = {
            "transform_id": "{}-{}".format(transform, id),
            "id": bug_info["id"],
            "bug_id": bug_info["bug_id"],
            "status": 0, # 0: not plausible, 1: plausible, 2: correct
            "diff": diff[id] if id in diff else None
        }
        if id in assessment:
            rs[id]["status"] = 2 if assessment[id] else 1
    return rs

def parse_origin_results(meta_path, annotation_path, tool):
    meta = json.load(meta_path)
    assessment, diff = parse_correctness_assessment(annotation_path, tool + "_original")
    rs = {}
    
    for bug_info in meta:
        id = bug_info["id"] 
        if len(bug_info["line_numbers"]) == 0:
            continue
        if bug_info["bug_id"] not in valid_bugs["original"]:
            continue
        rs[bug_info["bug_id"]] = {
            "status": 0, # 0: not plausible, 1: plausible, 2: correct
            "diff": diff[id] if id in diff else None
        }
        if id in assessment:
            rs[bug_info["bug_id"]]["status"] = 2 if assessment[id] else 1
    return rs

def evaluation_metrics(results, all_bugs):
    plausible_rate = 0
    correct_rate = 0
    
    for bug_id in results:
        if bug_id in all_bugs:
            if results[bug_id]["status"] >= 1:
                plausible_rate += 1
            if results[bug_id]["status"] == 2:
                correct_rate += 1
    total_bugs = len(list(all_bugs))
    return plausible_rate, correct_rate

def calculate_metrics(plausible, correct, all_bugs):
    plausible_rate = 0
    for bug_id in all_bugs:
        if bug_id in plausible:
            plausible_rate += plausible[bug_id]/all_bugs[bug_id]
    
    correct_rate = 0
    for bug_id in all_bugs:
        if bug_id in correct:
            correct_rate += correct[bug_id]/all_bugs[bug_id]
        
    total_bugs = len(list(all_bugs.keys()))
    return plausible_rate, correct_rate

def normalized_evaluation_metrics(results):
    plausible = {}
    correct = {}
    all_bugs = {}
    for idx in results:
        bug_id = results[idx]["bug_id"]
        all_bugs[bug_id] = all_bugs[bug_id] + 1 if bug_id in all_bugs else 1
        if results[idx]["status"] >= 1:
            plausible[bug_id] = plausible[bug_id] + 1 if bug_id in plausible else 1

        if results[idx]["status"] == 2:
            correct[bug_id] = correct[bug_id] + 1 if bug_id in correct else 1

    # calculate_metrics(plausible, correct, all_bugs)
   
    return plausible, correct, all_bugs

def unique_change_rate(positive_changes, negative_changes, all_bugs):
    unique_pos_change = len(list(positive_changes.keys()))
    unique_neg_change = len(list(negative_changes.keys()))
    unique_change = len(set(list(positive_changes.keys()) + list(negative_changes.keys())))
    total_bugs = len(list(all_bugs))
    return {
        "n_change": unique_change,
        "change_rate": round(unique_change*100/total_bugs, 1),
        "positive": unique_pos_change,
        "negative": unique_neg_change,
        "positive_rate": round(unique_pos_change*100/total_bugs, 1),
        "negative_rate": round(unique_neg_change*100/total_bugs, 1)
    }
    
def calculate_changes(transform_name, transform_results, original_results):
    positive_changes = defaultdict(list)
    negative_changes = defaultdict(list)
    all_bugs = {}
    transform_changes = []
    for idx in transform_results:
        status = transform_results[idx]["status"]
        bug_id = transform_results[idx]["bug_id"]
        all_bugs[bug_id] = all_bugs[bug_id] + 1 if bug_id in all_bugs else 1
        if status > original_results[bug_id]["status"]:
            positive_changes[bug_id].append(transform_results[idx]["transform_id"])
            transform_changes.append({
                "id": transform_results[idx]["id"],
                "diff": transform_results[idx]["diff"],
                "original_diff": original_results[bug_id]["diff"]
            })
        if status < original_results[bug_id]["status"]:
            negative_changes[bug_id].append(transform_results[idx]["transform_id"])
            transform_changes.append({
                "id": transform_results[idx]["id"],
                "diff": transform_results[idx]["diff"],
                "original_diff": original_results[bug_id]["diff"]
            })
    return positive_changes, negative_changes, all_bugs, transform_changes

def merge_dict(curr, new):
    for key in new:
        curr[key] = curr[key]  + new[key] if key in curr else  new[key]
    return curr

def evaluate(tool, transform, used_natural):
    original_results = parse_origin_results(open(f"data/meta/original.json", 'r'), f"data/plausible_patches/original-{tool}.xlsx", tool)
    all_transform_results = {}
    all_plausible, all_correct, all_bugs = {}, {}, {}
    all_positive_changes, all_negative_changes = {}, {}
    all_changes = []
    transform_results = parse_results(transform, open(f"data/meta/{transform}.json", 'r'), f"data/plausible_patches/{transform}-{tool}.xlsx", tool, used_natural)
    plausible, correct, bugs = normalized_evaluation_metrics(transform_results)
    print(correct)
    all_transform_results.update(transform_results)
    positive_changes, negative_changes, _, transform_changes = calculate_changes(transform, transform_results, original_results)
    all_plausible, all_correct, all_bugs = merge_dict(all_plausible, plausible), merge_dict(all_correct, correct), merge_dict(all_bugs, bugs)
    all_positive_changes, all_negative_changes = merge_dict(all_positive_changes, positive_changes), merge_dict(all_negative_changes, negative_changes)
    for change in transform_changes:
        all_changes.append(f'{transform}_{change["id"]}')
    
    return unique_change_rate(all_positive_changes, all_negative_changes, all_bugs)

def evaluate_all(tool, used_natural):
    f = open("data/changes/{}.txt".format(tool), "w")
    original_results = parse_origin_results(open(f"data/meta/original.json", 'r'), f"data/plausible_patches/original-{tool}.xlsx", tool)
    all_transform_results = {}
    all_plausible, all_correct, all_bugs = {}, {}, {}
    all_positive_changes, all_negative_changes = {}, {}
    all_changes = []
    for transform in ["naming", "expression", "statement"]:
        transform_results = parse_results(transform, open(f"data/meta/{transform}.json", 'r'), f"data/plausible_patches/{transform}-{tool}.xlsx", tool, used_natural)
        plausible, correct, bugs = normalized_evaluation_metrics(transform_results)
        all_transform_results.update(transform_results)
        positive_changes, negative_changes, _, transform_changes = calculate_changes(transform, transform_results, original_results)
        all_plausible, all_correct, all_bugs = merge_dict(all_plausible, plausible), merge_dict(all_correct, correct), merge_dict(all_bugs, bugs)
        all_positive_changes, all_negative_changes = merge_dict(all_positive_changes, positive_changes), merge_dict(all_negative_changes, negative_changes)
        for change in transform_changes:
            # if change["original_diff"] is None or change["diff"] is None:
            #     continue
            f.write('==============================\n')
            f.write(f'{transform}-{change["id"]}\n')
            all_changes.append(f'{transform}_{change["id"]}')
            f.write('------------------------------\n')
            if change["diff"] is not None:
                f.write(change["diff"] + "\n")
            else:
                f.write("None\n")
            f.write('------------------------------\n')
            if change["original_diff"] is not None:
                f.write(change["original_diff"] + "\n")
            else:
                f.write("None\n")

    plausible_rate_b, correct_rate_b = evaluation_metrics(original_results, all_bugs)
    evaluation_metrics(all_transform_results, list(all_transform_results.keys()))
    plausible_rate_a, correct_rate_a = calculate_metrics(all_plausible, all_correct, all_bugs)
    result = unique_change_rate(all_positive_changes, all_negative_changes, all_bugs)
    result["plausible_rate"] = plausible_rate_a
    result["correct_rate"] = correct_rate_a
    result["original_plausible_rate"] = plausible_rate_b
    result["original_correct_rate"] = correct_rate_b
    return result, all_positive_changes, all_negative_changes