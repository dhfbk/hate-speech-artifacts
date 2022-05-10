import argparse
import emoji
import math
import numpy as np
import os
import pandas as pd
import sys
import utils

from collections import Counter
from transformers import BertTokenizerFast


DATA_FOLDER = "data"
DATA_SUBSETS_FOLDER = "subsets"
RESULTS_FOLDER = os.path.join("results", "token-contributions")
SPECIAL_TOKENS = ["[USER]", "[URL]", "[EMAIL]"]
EMOJI_TOKEN = "[EMOJI]"
EMOJIS_TOKENS = list(emoji.UNICODE_EMOJI['en'].keys())
PUNCTUATION_TOKENS = ["!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", 
                      ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", 
                      "~", "…", "‘", "’", "“", "”"]
EPSILON = 1e-16

LABEL_MAPPING = {
    "founta": {
        "all-vs-hate": {0: 0, 1: 0, 2: 1},
        "all-vs-toxic": {0: 0, 1: 1, 2: 1}, # == no-vs-all
        "all-vs-abusive": {0: 0, 1: 1, 2: 0},
        "abusive-vs-hate": {1: 0, 2: 1},
        "no-vs-hate": {0: 0, 2: 1},
        "no-vs-abusive": {0: 0, 1: 1}
    },
    "gab": {
        "all-vs-hate": {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0},
        "all-vs-toxic": {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}, # == no-vs-all
        "all-vs-abusive": {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1},
        "abusive-vs-hate": {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0},
        "no-vs-hate": {0: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
        "no-vs-abusive": {0: 0, 1: 1, 8: 1}
    },
    "stormfront": {
        "all-vs-hate": {0: 0, 1: 1},
        "all-vs-toxic": {0: 0, 1: 1}, # == no-vs-all
        "all-vs-abusive": {0: 0, 1: 1},
        "abusive-vs-hate": {0: 0, 1: 1},
        "no-vs-hate": {0: 0, 1: 1},
        "no-vs-abusive": {0: 0, 1: 1}
    },
    "cad": {
        "all-vs-hate": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 0},
        "all-vs-toxic": {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}, # == no-vs-all
        "all-vs-abusive": {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 0, 10: 0, 11: 0, 12: 1},
        "abusive-vs-hate": {2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 0},
        "no-vs-hate": {0: 0, 1: 0, 4: 1, 5: 1, 6: 1, 9: 1, 10: 1, 11: 1},
        "no-vs-abusive": {0: 0, 1: 0, 2: 1, 3: 1, 7: 1, 8: 1, 12: 1}
    }
}


def compute_pmi(w_count, l_count, w_l_count, num_texts):
    """A function that computes pointwise mutual information between tokens and labels."""
    pmi_scores = {l:{} for l in l_count.keys()}

    for l in l_count.keys():
        for w in w_count.keys():
            # P(w): occurrences of "w" in texts over the total number of texts (across labels)
            p_w = w_count[w] / float(num_texts)

            if (w, l) in w_l_count.keys():
                # P(w|l): co-occurrences of "w" and "l" in texts over the number of texts with label l
                p_w_l = w_l_count[(w, l)] / float(l_count[l])

                # PMI(w,l) = P(w|l)/P(w): pointwise mutual information
                pmi = math.log2(p_w_l / float(p_w))

                # Adjustment factor; co-occurrences of "w" and "l" in texts
                adj_factor = w_l_count[(w, l)]

                # Reweighted PMI(w,l) = PMI(w,l)*adj_factor: reweighted PMI to account for low-frequency terms
                rpmi = pmi * adj_factor

                # Positive reweighted PMI(w,l): all values below 0 are normalized to EPSILON
                if rpmi <= 0.0: rpmi = EPSILON

                # Add the scores to the dictionary
                pmi_scores[l][w] = rpmi

    return pd.DataFrame(pmi_scores)

def read_tsv(input_filepath):
    """A function that returns the texts and labels corresponding to a given training input filepath."""
    texts, labels = [], []

    with open(input_filepath, "r") as f:
        for line in f:
            label, text = line.rstrip().split("\t")
            labels.append(label)
            texts.append(text)

    return texts, labels


def get_counts(texts, label, labels_str, tokenizer, tokenizer_type):
    """A function that returns token, label, and token-label counters after tokenizing the text 
    according to a given pretrained tokenizer, and normalizing the binary labels into strings."""
    token_counter, label_counter, token_label_counter = Counter(), Counter(), Counter()

    for i in range(len(texts)):
        curr_tokens = tokenizer.tokenize(texts[i])
        curr_label = labels_str[1] if label==1 else labels_str[0]

        label_counter[curr_label] += 1
        for curr_token in set(curr_tokens):
            # Retain all tokens except stopwords
            if curr_token not in utils.STOP_WORDS:
                if curr_token != "":
                    token_counter[curr_token] += 1
                    token_label_counter[(curr_token, curr_label)] += 1

    return token_counter, label_counter, token_label_counter

def normalize_pmi(pmi_scores):
    """A function that returns a normalized PMI dataframe in [0,1]."""

    def min_max_normalization(dataframe):
        """A function that performs the min-max normalization over log2 PMI scores."""
        df_normalized = dataframe.copy()

        for column in df_normalized.columns:
            curr_column = df_normalized[column]
            column_min = df_normalized[column].min()
            column_max = df_normalized[column].max()
            df_normalized[column] = (curr_column - column_min) / (column_max - column_min)
            
        return df_normalized

    # Fill missing values with epsilon for calculating the log2
    pmi_scores = pmi_scores.fillna(EPSILON)

    # Normalize log2 PMI values in [0,1] (flattening negative values to zero)
    pmi_scores = np.log2(pmi_scores)
    pmi_scores[pmi_scores < 0.0] = 0.0
    pmi_normalized = min_max_normalization(pmi_scores)

    return pmi_normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--dataset_name", type=str, required=False, default="founta", choices=["founta", "gab", "stormfront", "cad"], help="")
    parser.add_argument("-L", "--label_partitions", type=str, required=False, default="abusive-vs-hate", choices=["all-vs-hate", "all-vs-toxic", "all-vs-abusive", "abusive-vs-hate", "no-vs-hate", "no-vs-abusive"], help="")
    parser.add_argument("-R", "--region_type", type=str, required=False, default="all", choices=["all", "easy", "ambiguous", "hard"], help="")
    parser.add_argument("-P", "--region_percent", type=int, required=False, default=100, choices=[1, 5, 10, 17, 25, 33, 50, 75, 100], help="")
    parser.add_argument("-E", "--add_emojis", default=True, action="store_false", help="Whether or not adding emojis to the special token dictionary.")
    parser.add_argument("-T", "--pretrained_tokenizer", type=str, required=False, default="bert-base-uncased", choices=["bert-base-uncased"], help="")
    args = parser.parse_args()

    # Create input filepath and output filename prefix
    if args.region_type == "all":
        input_filename = f"{args.dataset_name}.train"
        input_filepath = os.path.join(DATA_FOLDER, input_filename)
        output_filename_prefix = f"pmi.{args.label_partitions}"
    else:
        input_filename = f"{args.dataset_name}.{args.region_type}-{args.region_percent}.train"
        input_filepath = os.path.join(DATA_FOLDER, DATA_SUBSETS_FOLDER, input_filename)
        output_filename_prefix = f"pmi.{args.label_partitions}.{args.region_percent}"

    # Ensure the output folders exist, o.w. create them
    output_folder = os.path.join(RESULTS_FOLDER, args.pretrained_tokenizer, f"{args.dataset_name}.{args.region_type}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get texts and associated labels from the original/subset input file
    input_filepath = os.path.join(DATA_FOLDER, args.dataset_name + "-mc" + ".train")
    texts, labels = read_tsv(input_filepath)

    # Create a dictionary of "label: [text1,...,textN]" elements to ease instance retrieval
    by_label = {}
    for i in range(len(labels)):
        if int(labels[i]) not in by_label.keys():
            by_label[int(labels[i])] = [texts[i]]
        else:
            by_label[int(labels[i])].append(texts[i])

    # Initialize the pretrained tokenizer with special tokens
    if args.pretrained_tokenizer == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_tokenizer)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_tokenizer)
    SPECIAL_TOKENS = (SPECIAL_TOKENS+EMOJIS_TOKENS) if (args.add_emojis == True) else (SPECIAL_TOKENS+[EMOJI_TOKEN])
    special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    # Tokenize text, normalize labels, and count token/label/token-label occurrences
    token_counter, label_counter, token_label_counter = Counter(), Counter(), Counter()
    labels_str = args.label_partitions.split("-vs-")
    for raw_label, label in LABEL_MAPPING[args.dataset_name][args.label_partitions].items():
        if raw_label in by_label:
            if label == 0:
                curr_token_counters, curr_label_counters, curr_token_label_counters = get_counts(
                    by_label[raw_label], label, labels_str, tokenizer, args.pretrained_tokenizer)
            else:
                curr_token_counters, curr_label_counters, curr_token_label_counters = get_counts(
                    by_label[raw_label], label, labels_str, tokenizer, args.pretrained_tokenizer)
            
            token_counter += curr_token_counters
            label_counter += curr_label_counters
            token_label_counter += curr_token_label_counters
        else:
            print(f"Skipping class {raw_label}: no examples in the training set.")
    
    # Get the total count of texts according to the labels that are taken into consideration
    texts_count = sum(label_counter.values())

    # Calculate pointwise mutual information
    pmi_scores = compute_pmi(token_counter, label_counter, token_label_counter, texts_count)

    # Normalize scores in [0,1] to allow cross-dataset operations
    pmi_scores_norm = normalize_pmi(pmi_scores)

    # Sort results by label (descending) and write results to files
    for label in labels_str:
        output_filename = f"{output_filename_prefix}.{label}.tsv"
        output_filepath = os.path.join(output_folder, output_filename)
        sorted_pmi_scores = pmi_scores_norm[label].to_frame().sort_values(by=[label], ascending=False)
        sorted_pmi_scores.to_csv(output_filepath, sep="\t", header=False)

