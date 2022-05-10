import argparse
import emoji
import os
import sys

from transformers import BertTokenizerFast


SPECIAL_TOKENS = ["[USER]", "[URL]", "[EMAIL]"]
EMOJI_TOKEN = "[EMOJI]"
EMOJIS_TOKENS = list(emoji.UNICODE_EMOJI['en'].keys())

def initialize_tokenizer(tokenizer_name, add_emojis):
    if tokenizer_name == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    
    extra_tokens = (SPECIAL_TOKENS+EMOJIS_TOKENS) if (add_emojis == True) else (SPECIAL_TOKENS+[EMOJI_TOKEN])
    special_tokens_dict = {'additional_special_tokens': extra_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer

def get_artifacts(artifacts_filepath):
    artifacts_list = []
    with open(artifacts_filepath, "r") as f:
        for line in f:
            artifacts_list.append(line.rstrip())
    return artifacts_list

def apply_strategy_to_token(token, original_token, artifacts_list, strategy, tokenizer_name):
    if token in artifacts_list:
        if strategy == "del":
            return " ", True # this avoids creating new undesired words that could change the splitting at training time
        elif strategy == "mask":
            return "[unused0]", True
        elif strategy == "none":
            return original_token, True
        else:
            sys.exit(f"Strategy {strategy} does not exist.")
    else:
        return original_token, False

def create_data_variant(input_filepath, output_filepath, artifacts_filepath, 
    strategy, classes, tokenizer_name, add_emojis):
    artifacts_list = get_artifacts(artifacts_filepath)
    tokenizer = initialize_tokenizer(tokenizer_name, add_emojis)

    # Process the original file line by line, and write results to the output file
    output_file = open(output_filepath, "w")
    with open(input_filepath, "r") as f:
        for line in f:
            line = line.rstrip().split("\t")
            label = line[0]
            text = line[1]

            curr_tokens = tokenizer.tokenize(text, return_offsets_mapping=True)
            offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"][1:-1]

            line_has_token = False
            text_cleaned, curr_has_token = apply_strategy_to_token(curr_tokens[0], text[int(offsets[0][0]):int(offsets[0][1])], 
                artifacts_list, strategy, tokenizer_name)
            if curr_has_token == True:
                line_has_token = True
            for i in range(1, len(curr_tokens)):
                num_prev_spaces = offsets[i][0] - offsets[i-1][1]
                temp, curr_has_token = apply_strategy_to_token(
                    curr_tokens[i], text[int(offsets[i][0]):int(offsets[i][1])], artifacts_list, strategy, tokenizer_name)
                text_cleaned += (" " * num_prev_spaces) + temp
                if curr_has_token == True:
                    line_has_token = True

            if line_has_token == True:
                output_file.write(label + "\t" + text_cleaned + "\n")
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input_filepath", type=str, required=True, 
        help="The path to the input filepath.")
    parser.add_argument("-O", "--output_filepath", type=str, required=True, 
        help="The path to the output filepath.")
    parser.add_argument("-A", "--artifacts_filepath", type=str, required=True, 
        help="The artifacts to use.")
    parser.add_argument("-S", "--strategy", type=str, required=True, 
        choices=["none", "mask", "del"], help="The strategy to apply to sensitive tokens.")
    parser.add_argument("-C", "--classes", type=str, required=True, 
        choices=["all", "hateful"], help="Whether to restrict the masking/removal to some classes only.")
    parser.add_argument("-T", "--pretrained_tokenizer", type=str, required=False, 
        default="bert-base-uncased", choices=["bert-base-uncased"], help="")
    parser.add_argument("-E", "--add_emojis", default=True, action="store_false", 
        help="Whether or not adding emojis to the special token dictionary.")
    args = parser.parse_args()

    create_data_variant(args.input_filepath, args.output_filepath, args.artifacts_filepath, 
        args.strategy, args.classes, args.pretrained_tokenizer, args.add_emojis)
