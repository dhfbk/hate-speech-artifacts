import argparse
import numpy as np


def compute_score(input_base_filepath, target_dataset, number_of_seeds):
    scores = []
    output_filepath = input_base_filepath + "." + target_dataset + ".results"
    output_file = open(output_filepath, "w")

    for i in range(number_of_seeds):
        complete_input_filepath = input_base_filepath + "." + str(i+1) + "." + target_dataset + ".out.eval"
        with open(complete_input_filepath) as f:
            for line in f:
                first_key_value = line.rstrip().split(",")[0]
                score = float(first_key_value.split(": ")[1])
                scores.append(score)
                output_file.write("SEED #" + str(i+1) + ": " + str(score) + "\n")
                break

    output_file.write("\n")

    mean = round(np.mean(scores)*100, 2)
    stddev = round(np.std(scores, ddof=1)*100, 1)
    output_file.write("Avg score: " + str(mean) + "\n")
    output_file.write("Std score: " + str(stddev) + "\n")
    output_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input_base_filepath", type=str, required=True, 
        help="The path to the raw input folder.")
    parser.add_argument("-T", "--target_dataset", type=str, required=True, 
        help="The target dataset name.")
    parser.add_argument("-N", "--number_of_seeds", type=int, required=True,
        help="The number of seeds.")
    args = parser.parse_args()

    compute_score(args.input_base_filepath, args.target_dataset, args.number_of_seeds)
