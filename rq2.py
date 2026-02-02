import json
import argparse
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import matplotlib.pyplot as plt
from krippendorff import alpha
import os
import pandas as pd
from scipy import stats
from prettytable import PrettyTable

def agreement(data):
    print(data.shape)
    data = data.transpose()
    ratings_array, _ = aggregate_raters(data, n_cat=2)
    kappa = fleiss_kappa(ratings_array)
    return kappa

def MWW(A, B):
    # Ensure both arrays are NumPy arrays
    A = np.array(A)
    B = np.array(B)

    statistic, p_value = stats.mannwhitneyu(A, B, alternative='less')
    r = 1 - 2*statistic/(A.size*B.size)


    return statistic, p_value, r

def box_plot(data, labels, name):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a box plot for each array
    ax.boxplot(data, labels=labels)

    # Set labels and title
    ax.set_xlabel('Data Sets')
    ax.set_ylabel('Values')
    ax.set_title('Box Plot of Three Arrays')

    # Show the plot
    plt.savefig(name)
    
def write_ids_to_file(ids, file_name):
    with open(os.path.join("data/ids", file_name), "w") as f:
        f.write("\n".join(ids))
    
def calculate_naturalness(data):
    return data[data >=4], data[data == 3], data[data == 2], data[data <=1]

def main():
    # Specify the file name to read
    file_name = "data/human_study/survey/survey.json"

    # Read the JSON data from the file and load it into a dictionary
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    valid_bugs = {}
    for transform in ["naming", "expression", "statement"]:
        with open(f"data/valid_bugs/{transform}.txt", "r") as f:
            valid_bugs[transform] = []
            for line in f:
                valid_bugs[transform].append(int(line))

    with open("data/valid_bugs/original.txt", "r") as f:
        valid_bugs["original"] = f.read().split("\n")

    final = [[],[],[],[],[]]
    bug_info = []
    average_cc_score = []
    average_cr_score = []
    time = [[],[],[],[],[]]
    for level in data:
        for bug in data[level]:
            if int(bug) not in valid_bugs[level]:
                continue
            bug_info.append("{}_{}".format(level, bug))
            cc_score = 0
            cr_score = 0
            for participant in data[level][bug]:
                cur_final = 0
                cc_score += data[level][bug][participant]["CC"]
                cr_score += data[level][bug][participant]["CR"]
                if data[level][bug][participant]["CC"] >= 3:
                    cur_final = 1
                if data[level][bug][participant]["CR"] >= 3:
                    cur_final = 1
                ID = int(participant[1:]) - 1
                
                if ID >=5:
                    ID = ID - 5
                assert 0 <=ID and ID <=9
                final[ID].append(cur_final)
                time[ID].append(float(data[level][bug][participant]["Time"]))
            average_cc_score.append(cc_score/5)
            average_cr_score.append(cr_score/5)
    
    with(open("data/score/average_score.txt", "w")) as f:
        for idx, bug in enumerate(bug_info):
            f.write("{}\t{}\t{}\n".format(bug, average_cc_score[idx], average_cr_score[idx]))
    
    naming_range = np.array(range(854))
    expression_range = np.array(range(854, 994))
    statement_range = np.array(range(994, 1098))
    
    final = np.array(final)
    df = pd.DataFrame(final.transpose())
    df.to_csv("data/human_study/survey/survey.csv", index=False)
    results = final.sum(axis=0)
    time = np.array(time)
    bug_info = np.array(bug_info)

    # Some assertion
    for bug in bug_info[naming_range]:
        assert bug.startswith("naming")

    for bug in bug_info[expression_range]:
        assert bug.startswith("expression")
    
    for bug in bug_info[statement_range]:
        assert bug.startswith("statement")
        
    print("----------------------------------------------------------------------")
    print("==> [RQ2.1 - Section 5.1.1] Completion Time")
    _99th_percentile_time = np.percentile(time, 99)
    print("=> Overal")
    print("+ 99% Percentile: " + str(round(_99th_percentile_time, 2)))
    print("+ Average: " + str(np.mean(time[time < _99th_percentile_time])))
    print("+ STD: " +str(np.std(time[time < _99th_percentile_time])))
    
    
    naming_time = time[:, naming_range]
    naming_time = naming_time[naming_time < _99th_percentile_time]
    expression_time = time[:, expression_range]
    expression_time = expression_time[expression_time < _99th_percentile_time]
    statement_time = time[:, statement_range]
    statement_time = statement_time[statement_time < _99th_percentile_time]
    
    print("=> Table 11: Time taken by different transformation levels")
    table_11 = PrettyTable()
    table_11.field_names = ['Transformation Level', 'Average Time', 'STD']
    table_11.add_row(['Naming', np.mean(naming_time), np.std(naming_time)])
    table_11.add_row(['Expression', np.mean(expression_time), np.std(expression_time)])
    table_11.add_row(['Statement', np.mean(statement_time), np.std(statement_time)])
    print(table_11)

    print("=> Table 12: Time taken by high agreement vs high disgareement cases")
    natural_ids = np.where(results <=1)[0]
    write_ids_to_file(bug_info[natural_ids], "natural_ids.txt")
    unnatural_ids = np.where(results >=4)[0]
    write_ids_to_file(bug_info[unnatural_ids], "unnatural_ids.txt")
    likely_natural_ids = np.where(results == 2)[0]
    write_ids_to_file(bug_info[likely_natural_ids], "likely_natural_ids.txt")
    likely_unnatural_ids = np.where(results == 3)[0]
    write_ids_to_file(bug_info[likely_unnatural_ids], "likely_unnatural_ids.txt")
    agreement_ins = np.concatenate([natural_ids, unnatural_ids])
    write_ids_to_file(bug_info[agreement_ins], "agreement_ins.txt")
    disagrement_ids =  np.concatenate([likely_natural_ids, likely_unnatural_ids])
    write_ids_to_file(bug_info[disagrement_ids], "disagrement_ids.txt")

    agreement_time = time[:, agreement_ins]
    agreement_time = agreement_time[agreement_time < _99th_percentile_time]

    disagreement_time = time[:, disagrement_ids]
    disagreement_time = disagreement_time[disagreement_time < _99th_percentile_time]
    
    table_12 = PrettyTable()
    table_12.field_names = ['Category', 'Average Time', 'STD']
    table_12.add_row(['Agreement', np.mean(agreement_time), np.std(agreement_time)])
    table_12.add_row(['Disagreement', np.mean(disagreement_time), np.std(disagreement_time)])
    print(table_12)
    
    print("----------------------------------------------------------------------")
    print("==> [RQ2.2 - Section 5.1.2] Agreement")
    print("=> Table 13: Agreement Levels of Participants.")
    
    table_13 = PrettyTable()
    table_13.field_names = ['Agreement Level', 'Natural', 'Unnatural', 'Total']
    
    full_agreement_natural = len(results[results == 0])
    full_agreement_unnatural = len(results[results == 5])
    full_agreement_total = full_agreement_natural + full_agreement_unnatural
    
    high_agreement_natural = len(results[results <= 1]) 
    high_agreement_unnatural = len(results[results >= 4]) 
    high_agreement_total = high_agreement_natural + high_agreement_unnatural
    
    disagreement_natural = len(results[results == 2])
    disagreement_unnatural = len(results[results == 3])
    disagreement_total = disagreement_natural + disagreement_unnatural
    
    table_13.add_row(['Full Agreement', full_agreement_natural, full_agreement_unnatural, full_agreement_total])
    table_13.add_row(['High Agreement', high_agreement_natural, high_agreement_unnatural, high_agreement_total])
    table_13.add_row(['Disagreement', disagreement_natural, disagreement_unnatural, disagreement_total])
    print(table_13)
    
    print("----------------------------------------------------------------------")
    print("==> [RQ2.3 - Section 5.1.4] Naturalness Ratio")
    naming_naturalness = calculate_naturalness(results[naming_range])
    expression_naturalness = calculate_naturalness(results[expression_range])
    statemet_naturalness = calculate_naturalness(results[statement_range])
    full_naturalness = calculate_naturalness(results)
    categories = ['Naming', 'Expression', 'Statement', 'Overall']

    naming_values = np.array([len(i) for i in naming_naturalness])
    expression_values =  np.array([len(i) for i in expression_naturalness])
    statement_values =  np.array([len(i) for i in statemet_naturalness])
    full_values =  np.array([len(i) for i in full_naturalness])

    print("=> Table 14: Naturalness Ratio")
    table_14 = PrettyTable()
    table_14.field_names = ['Level', 'Highly Natural (>=4)', 'Neutral (3)', 'Neutral (2)', 'Highly Unnatural (<=1)']
    table_14.add_row(['Naming', naming_values[0], naming_values[1], naming_values[2], naming_values[3]])
    table_14.add_row(['Expression', expression_values[0], expression_values[1], expression_values[2], expression_values[3]])
    table_14.add_row(['Statement', statement_values[0], statement_values[1], statement_values[2], statement_values[3]])
    table_14.add_row(['Overall', full_values[0], full_values[1], full_values[2], full_values[3]])
    print(table_14)
    
    assert naming_values.sum() == 854
    assert expression_values.sum() == 140
    assert statement_values.sum() == 104
    assert full_values.sum() == 1098

if __name__ == "__main__":
    main()