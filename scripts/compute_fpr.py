import argparse
import sys
from sklearn.metrics import confusion_matrix


def compute_fpr_score(input_pred_filepath, input_gold_filepath, output_filepath):
    count = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    gold_labels = []
    pred_labels = []

    with open(input_gold_filepath, "r") as f:
        for line in f:
            label = int(line.rstrip().split("\t")[0])
            gold_labels.append(label)

    with open(input_pred_filepath, "r") as f:
        for line in f:
            label = int(line.rstrip().split("\t")[0])
            pred_labels.append(label)

    for i in range(len(gold_labels)):
        if gold_labels[i] == pred_labels[i]:
            if gold_labels[i] == 1:
                tp += 1
            elif gold_labels[i] == 0:
                tn += 1
            else:
                sys.exit("Error. Exiting")
        elif gold_labels[i] != pred_labels[i]:
            if gold_labels[i] == 1:
                fn += 1
            elif gold_labels[i] == 0:
                fp += 1
            else:
                sys.exit("Error. Exiting")
        else:
            sys.exit("Error. Exiting")
        
        count += 1

    output_file = open(output_filepath, "w")
    score_string = str(fp / (fp+tn))
    results_string = "{'.run/.sum': " + score_string + ",\n '.run/hatelabels/fpr': " + score_string + "}"
    output_file.write(results_string)
    output_file.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--input_pred_filepath", type=str, required=True, 
        help="")
    parser.add_argument("-G", "--input_gold_filepath", type=str, required=True, 
        help="")
    parser.add_argument("-O", "--output_filepath", type=str, required=True, 
        help="The path to the output filepath.")
    args = parser.parse_args()
    
    compute_fpr_score(args.input_pred_filepath, args.input_gold_filepath, args.output_filepath)
