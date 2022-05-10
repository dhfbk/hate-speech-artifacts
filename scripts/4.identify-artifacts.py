import argparse
import emoji
import os
import pandas as pd
import random
import utils

from transformers import BertTokenizerFast

RANDOM_SEED = 42
INPUT_BASE_FILEPATH = os.path.join("results", "token-contributions")
DATASET_NAMES = ["founta", "gab", "stormfront", "cad"]
SPECIAL_TOKENS = ["[USER]", "[URL]", "[EMAIL]"]
EMOJI_TOKEN = "[EMOJI]"
EMOJIS_TOKENS = list(emoji.UNICODE_EMOJI['en'].keys())
PUNCTUATION_TOKENS = ["!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", 
                      ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", 
                      "~", "…", "‘", "’", "“", "”"]


def get_input_filepaths(partitions, curr_partition, tokenizer_name):
    input_filepaths = []
    for dataset_name in DATASET_NAMES:
        filepath = os.path.join(INPUT_BASE_FILEPATH, tokenizer_name, 
            dataset_name + ".all", "pmi." + partitions + "." + curr_partition + ".tsv")
        input_filepaths.append(filepath)
    return input_filepaths

def add_tokens_to_index(token_index, tokens):
    for t in set(tokens):
        # Retain all tokens except stopwords
        if t not in utils.STOP_WORDS:
            if t != "":
                if t in token_index.keys():
                    token_index[t].append(tokens)
                else:
                    token_index[t] = [tokens]

    return token_index


def create_token_index(curr_partition, tokenizer, tokenizer_name):
    token_index = dict()

    # Iterate over the datasets and store examples for each relevant token
    for dataset_name in DATASET_NAMES:
        with open(os.path.join("data", dataset_name + ".train"), "r") as f:
            for line in f:
                label, text = line.rstrip().split("\t")
                if curr_partition == "all":
                    selected_label = "0"
                elif curr_partition == "hate":
                    selected_label = "1"
                else:
                    sys.exit(f"Unrecognized {curr_partition} partition.")
                
                if label == selected_label:
                    curr_tokens = tokenizer.tokenize(text)

                    # Store token-text index for further checking / validation / annotation
                    token_index = add_tokens_to_index(token_index, curr_tokens)

    return token_index

def get_examples_for_token(token_index, token, num_examples):
    if (token_index is not None) and (token in token_index):
        random.Random(RANDOM_SEED).shuffle(token_index[token])
        if len(token_index[token]) > num_examples:
            return token_index[token][:num_examples]
        else:
            return token_index[token][:len(token_index[token])]
    else:
        return []

def initialize_tokenizer(tokenizer_name, add_emojis):
    if tokenizer_name == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    
    extra_tokens = (SPECIAL_TOKENS+EMOJIS_TOKENS) if (add_emojis == True) else (SPECIAL_TOKENS+[EMOJI_TOKEN])
    special_tokens_dict = {'additional_special_tokens': extra_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer

def compute_inter_corpora_contributions(input_filepaths, partitions, curr_partition, tokenizer_name):
    def read_token_contributions(input_filepath, corpus_name):
        header_list = ["token", corpus_name]
        corpus_tokens = pd.read_csv(input_filepath, sep="\t", names=header_list, lineterminator="\n")
        return corpus_tokens

    # Collect the token contribution dataframe for each corpus
    contribution_dfs = []
    for i in range(len(input_filepaths)):
        corpus_tokens = read_token_contributions(input_filepaths[i], DATASET_NAMES[i])
        contribution_dfs.append(corpus_tokens)
    
    # Join all dataframes to the first one
    contributions = contribution_dfs[0].set_index('token').join(
        contribution_dfs[1].set_index('token')).join(
        contribution_dfs[2].set_index('token')).join(
        contribution_dfs[3].set_index('token'))
    
    # Fill OOV (NA values) with 0.0
    contributions = contributions.fillna(0.0)

    # Add descriptive statistics columns
    contributions['mean'] = contributions.mean(axis=1)
    contributions['std'] = contributions.std(axis=1)

    # Sort the resulting dataframe by mean
    contributions = contributions.sort_values(by=["mean"], ascending=False)

    # Ensure the output folder exists, o.w. create it
    output_folder = os.path.join(INPUT_BASE_FILEPATH, tokenizer_name, "summary")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Print the inter-corpora token contributions to an output file in the output folder
    output_filepath = os.path.join(output_folder, "pmi." + partitions + "." + curr_partition + ".tsv")
    contributions.to_csv(output_filepath, sep="\t", header=True)

    return output_folder, output_filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--label_partitions", type=str, required=False, default="abusive-vs-hate", choices=["all-vs-hate", "all-vs-toxic", "all-vs-abusive", "abusive-vs-hate", "no-vs-hate", "no-vs-abusive"], help="")
    parser.add_argument("-T", "--pretrained_tokenizer", type=str, required=False, default="bert-base-uncased", choices=["bert-base-uncased"], help="")
    parser.add_argument("-E", "--add_emojis", default=True, action="store_false", help="Whether or not adding emojis to the special token dictionary.")
    parser.add_argument("-I", "--create_index", default=True, action="store_false", help="Whether or not creating the token-text index.")
    parser.add_argument("-S", "--threshold", type=float, required=False, default=0.3, choices=[0.3], help="")
    args = parser.parse_args()

    # Initialize the tokenizer if a token-text index needs to be built
    if args.create_index == True:
        tokenizer = initialize_tokenizer(args.pretrained_tokenizer, args.add_emojis)

    # Process each partition (e.g., hateful vs abusive) specified as command line argument
    partitions = args.label_partitions.split("-vs-")
    for curr_partition in partitions:
        # Collect filepaths to intra-corpora token contributions
        input_filepaths = get_input_filepaths(
            args.label_partitions, curr_partition, args.pretrained_tokenizer)

        # Compute the inter-corpora token statistics and return the output folder and filepath
        output_folder, contributions_filepath = compute_inter_corpora_contributions(
            input_filepaths, args.label_partitions, curr_partition, args.pretrained_tokenizer)

        # If required, create the token-text index to provide in-context examples of token occurrences
        # The examples are collected regardless of the origin dataset on purpose        
        # @TODO: Limited to all-vs-hate only as for now, eventually extend it
        token_index = None
        if (args.create_index == True) and (args.label_partitions == "all-vs-hate"):
            token_index = create_token_index(curr_partition, tokenizer, args.pretrained_tokenizer)

        # Subset the token contributions to an output file in the output folder
        # The subset will contain all tokens whose value is greater or equal to a threshold given as input
        # If examples for the tokens have been collected, a column for each one will be appended
        output_filepath_avg = os.path.join(
            output_folder, "avg." + args.label_partitions + "." + curr_partition + ".tsv")
        output_file_avg = open(output_filepath_avg, "w")

        is_header = True
        max_examples = 10

        for line in open(contributions_filepath, "r"):
            if is_header:
                output_file_avg.write(line.rstrip())
                # @TODO: Limited to all-vs-hate only as for now, eventually extend it
                if (args.create_index == True) and (args.label_partitions == "all-vs-hate"):
                    for index in range(max_examples):
                        output_file_avg.write("\texample-" + str(index+1))
                output_file_avg.write("\n")
                is_header = False

            else:
                token, founta, gab, stormfront, cad, mean, std = line.split("\t")
                if float(mean) >= args.threshold:
                    examples_string = ""
                    examples = get_examples_for_token(token_index, token, max_examples)

                    for example in examples:
                        examples_string += "\t"
                        for curr_token in example:
                            if curr_token.startswith("##"):
                                examples_string += curr_token
                            else:
                                examples_string += " " + curr_token
                    output_file_avg.write(line.rstrip() + examples_string + "\n")

        output_file_avg.close()