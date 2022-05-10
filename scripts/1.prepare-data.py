import argparse
import csv
import demoji
import logging
import numpy as np
import os
import pandas as pd
import random
import re
import shutil
import string
import sys
import wordsegment as ws

from collections import Counter
from html import unescape
from sklearn.model_selection import StratifiedKFold

ws.load()   # load the vocabulary for wordsegment


RANDOM_SEED = 42
AMBIGUOUS_LABEL_NAME = "ambiguous"

EXTRA_LABELS = {
    "founta": ["spam"], 
    "gab": [], 
    "stormfront": ["relation", "idk/skip"],
    "cad": []}
LABEL_TO_ID = {
    "founta": {"normal": 0, "abusive": 1, "hateful": 2, "spam": 3},
    "gab": {"noHate": 0, "vo": 1, "hd": 2, "cv": 3, "hd$cv": 4, "hd$vo": 5, "cv$vo": 6, "hd$cv$vo": 7, "+ABUS": 8},
    "stormfront": {"noHate": 0, "hate": 1, "relation": 2, "idk/skip": 3},
    "cad": {"Neutral": 0, "Slur": 1, "PersonDirectedAbuse": 2, "AffiliationDirectedAbuse": 3, "IdentityDirectedAbuse": 4, "AffiliationDirectedAbuse$IdentityDirectedAbuse": 5, "AffiliationDirectedAbuse$IdentityDirectedAbuse$PersonDirectedAbuse": 6, "AffiliationDirectedAbuse$PersonDirectedAbuse": 7, "AffiliationDirectedAbuse$Slur": 8, "IdentityDirectedAbuse$PersonDirectedAbuse": 9, "IdentityDirectedAbuse$PersonDirectedAbuse$Slur": 10, "IdentityDirectedAbuse$Slur": 11, "PersonDirectedAbuse$Slur": 12}}
ID_TO_LABEL = {
    "founta": {0: "normal", 1: "abusive", 2: "hateful", 3: "spam"},
    "gab": {0: "noHate", 1: "vo", 2: "hd", 3: "cv", 4: "hd$cv", 5: "hd$vo", 6: "cv$vo", 7: "hd$cv$vo", 8: "+ABUS"},
    "stormfront": {0: "noHate", 1: "hate", 2: "relation", 3: "idk/skip"},
    "cad": {0: "Neutral", 1: "Slur", 2: "PersonDirectedAbuse", 3: "AffiliationDirectedAbuse", 4: "IdentityDirectedAbuse", 5: "AffiliationDirectedAbuse$IdentityDirectedAbuse", 6: "AffiliationDirectedAbuse$IdentityDirectedAbuse$PersonDirectedAbuse", 7: "AffiliationDirectedAbuse$PersonDirectedAbuse", 8: "AffiliationDirectedAbuse$Slur", 9: "IdentityDirectedAbuse$PersonDirectedAbuse", 10: "IdentityDirectedAbuse$PersonDirectedAbuse$Slur", 11: "IdentityDirectedAbuse$Slur", 12: "PersonDirectedAbuse$Slur"}}
LABEL_MAPPING = {
    "founta": {0: 0, 1: 0, 2: 1, 3: 0},
    "gab": {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0},
    "stormfront": {0: 0, 1: 1, 2: 0, 3: 0},
    "cad": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 0}}
ABUSIVE_IDS = {
    "founta": [1],
    "gab": [1, 8],
    "stormfront": [],
    "cad": [2, 3, 7, 8, 12]
}


def create_dataset_file(input_folder, dataset_name):

    def create_id_to_text(docs_folder):
        id_to_text = dict()

        for filename in os.listdir(docs_folder):
            if os.path.isfile(os.path.join(docs_folder, filename)):
                file_id = filename.rstrip(".txt")

                with open(os.path.join(docs_folder, filename), "r") as f:
                    text = f.read()
                    if file_id not in id_to_text:
                        id_to_text[file_id] = text.rstrip()
                    else:
                        sys.exit(f"File ID {file_id} is already in the id_to_text dict.")

        return id_to_text

    def is_gab_target_protected(text, target_values):
        targets = Counter()

        for label, values in target_values.items():
            if label != "annotations":
                values = ["0.0" if element == "" else element for element in values]
                binary_ann = Counter(values)
                if "1.0" in binary_ann.keys():
                    targets[label] = binary_ann["1.0"]

        # Cases without any target, which should be abusive instead of hateful (e.g., id: 203)
        if len(targets) == 0:
            # print(f"Without target:\n\ttext: {text}\n\tannotations: {target_values}")
            return False # 22
        
        # Cases where the target is POL or IDL
        most_common, most_common_count = Counter(targets).most_common(1)[0]
        if most_common in ["POL", "IDL"]:
            # print(f"With POL/IDL target:\n\ttext: {text}\n\tannotations: {target_values}")
            return False # 627
        else:
            return True

    if dataset_name == "founta":
        # Get the paths to the relevant files/folders of the dataset
        annotations_filepath = os.path.join(input_folder, "hatespeech_text_label_vote_RESTRICTED_100K.csv")

        # Create the dataset file while reading the annotation file
        output_file = open(os.path.join(input_folder, "data.tsv"), "a")
        with open(annotations_filepath, "r") as f:
            for line in f:
                raw_text, raw_label, _ = line.rstrip().split("\t")
                output_file.write(raw_label + "\t" + raw_text + "\n")
        output_file.close()

    elif dataset_name == "gab":
        # Get the paths to the relevant files/folders of the dataset
        annotations_filepath_a = os.path.join(input_folder, "ghc_train.tsv")
        annotations_filepath_b = os.path.join(input_folder, "ghc_test.tsv")
        target_annotations_filepath = os.path.join(input_folder, "GabHateCorpus_annotations.tsv")

        # Store raw target annotations to exclude POL and IDL later
        post_targets = dict()
        with open(target_annotations_filepath, "r") as f:
            is_header = True    # to skip the header

            for line in f:
                if is_header:
                    is_header = False
                else:
                    line = line.split("\t")
                    text = line[2]

                    if text not in post_targets:
                        post_targets[text] = {
                            "REL": [line[7]], "RAE": [line[8]], "SXO": [line[9]], "GEN": [line[10]], 
                            "IDL": [line[11]], "NAT": [line[12]], "POL": [line[13]], "MPH": [line[14]]}
                    else:
                        post_targets[text]["REL"].append(line[7])
                        post_targets[text]["RAE"].append(line[8])
                        post_targets[text]["SXO"].append(line[9])
                        post_targets[text]["GEN"].append(line[10])
                        post_targets[text]["IDL"].append(line[11])
                        post_targets[text]["NAT"].append(line[12])
                        post_targets[text]["POL"].append(line[13])
                        post_targets[text]["MPH"].append(line[14])

        # Create the dataset file while reading the annotation files
        output_file = open(os.path.join(input_folder, "data.tsv"), "a")    
        for curr_filepath in [annotations_filepath_a, annotations_filepath_b]:
            with open(curr_filepath, "r") as f:
                is_header = True    # to skip the header

                for line in f:
                    if is_header:
                        is_header = False
                    else:
                        raw_label = ""
                        raw_text, hd, cv, vo = line.rstrip().split("\t")

                        # Create a possibly composite label
                        if hd == "1": 
                            raw_label += "hd"
                        if cv == "1":
                            if len(raw_label) != 0: raw_label += "$"
                            raw_label += "cv"
                        if vo == "1":
                            if len(raw_label) != 0: raw_label += "$"
                            raw_label += "vo"
                        if ((hd == "0") and (cv == "0") and (vo == "0")) and (len(raw_label)==0):
                            raw_label += "noHate"

                        if (hd == "1") or (cv == "1"):
                            if not is_gab_target_protected(raw_text, post_targets[raw_text]):
                                raw_label = "+ABUS"

                        output_file.write(raw_label + "\t" + raw_text + "\n")

        output_file.close()

    elif dataset_name == "stormfront":
        # Get the paths to the relevant files/folders of the dataset
        annotations_filepath = os.path.join(input_folder, "annotations_metadata.csv")
        docs_folder = os.path.join(input_folder, "all_files")

        # Create a dictionary storing the content of each file associated with its file ID
        id_to_text = create_id_to_text(docs_folder)

        # Create the dataset file by using the id_to_text dictionary while reading the annotation file
        output_file = open(os.path.join(input_folder, "data.tsv"), "w")
        with open(annotations_filepath, "r") as f:
            is_header = True    # to skip the header

            # Get content from the dictionary and write it along with its label to the dataset file
            for line in f:
                if is_header:
                    is_header = False
                else:
                    file_id, _, _, _, raw_label = line.rstrip().split(",")
                    output_file.write(raw_label + "\t" + id_to_text[file_id] + "\n")

        output_file.close()

    elif dataset_name == "cad":
        # Get the paths to the relevant files/folders of the dataset
        annotations_filepath = os.path.join(input_folder, "cad_v1_1.tsv")

        # Load the dataset into a dataframe and keep entries that are not marked as excluded and do not require previous content
        df = pd.read_csv(annotations_filepath, 
            delimiter="\t", quoting=csv.QUOTE_NONE, keep_default_na=False)
        df = df.loc[df['split'].isin(['train', 'dev', 'test'])]
        df = df.loc[df['annotation_Context'].isin(['CurrentContent', '"NA"'])]

        # Remove row with empty "meta_text" cell (id: cad_11730)
        df = df[df.id != "cad_11730"]

        # Merge rows corresponding to the same post keeping track of different labels (multi-label annotation, not disagreement)
        static_cols = ['info_id', 'info_subreddit', 'info_subreddit_id', 'info_id.parent',
            'info_id.link', 'info_thread.id', 'info_order', 'info_image.saved',
            'meta_author', 'meta_created_utc', 'meta_date', 'meta_day', 'meta_permalink', 'meta_text']
        df = df.groupby(static_cols)["annotation_Primary"].apply(lambda x: '$'.join(sorted(set(x)))).reset_index()

        # Store relevant column series
        texts = df.meta_text.astype("string").tolist()
        labels = df.annotation_Primary.astype("string").tolist()

        # Create the dataset by keeping relevant column series (and labels=col0, tests=col1)
        output_file = open(os.path.join(input_folder, "data.tsv"), "w")
        for i in range(len(texts)):
            curr_text = texts[i].replace("\\\\", "[DOUBLE_BS]").replace("\\", "").replace("[DOUBLE_BS]", "\\")
            output_file.write(str(labels[i]) + "\t" + curr_text + "\n")
        output_file.close()

    else:
        sys.exit(f"Dataset {dataset_name} is not handled.")


def clean_text(text, dataset_name, keep_emojis):
    
    def convert_emojis_to_text(text):
        matches = demoji.findall_list(text, desc=False)
        text = demoji.replace(text, repl=" [EMOJI] ")
        return text

    def regex_match_segmentation(match):
        # Useful to segment hashtags found via regexes
        return ' '.join(ws.segment(match.group(0)))

    def replace_cad_urls(text):
        # based on https://github.com/dongpng/cad_naacl2021/blob/main/src/contextual_abuse_dataset.py
        text = re.sub(r"\[([^\[\]]+)\]\((https:\/\/(.*?))\)", r"\1", text)
        text = re.sub(r"\[([^\[\]]+)\]\((\/message\/compose(.*?))\)", r"\1", text)
        text = re.sub(r"\[([^\[\]]+)\]\((\/r\/(.*?))\)", r"\1", text)
        text = re.sub(r'http(s?):\/\/[^\r\n\t\f\v )\]\}]+', '[URL]', text) # excludes trailing parentheses too
        text = re.sub(r'www\.\S+', '[URL]', text)
        return text

    text = unescape(text)   # HTML tags handling
    text = text.lower()     # make it lowercase

    # Normalize most common space-split URLs (for noisy Stormfront data)
    if dataset_name == "stormfront":
        text = text.replace("http : //", "http://")
        text = text.replace("https : //", "https://")
        for special_char in ["_", "=", "&", "?", "%", "(", ")", "..."]:
            text = text.replace(" " + special_char + " ", special_char)
    
    # Replace email addresses
    text = re.sub(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", text)

    # Replace URLs and user/channel mentions
    if (dataset_name == "cad"):
        text = replace_cad_urls(text) # handles composite [string](url) on Reddit
        text = re.sub(r"\/u\/\w+", "[USER]", text) # /u/user on Reddit only
        text = re.sub(r"\/r\/\w+", "[USER]", text) # /r/subreddit on Reddit only
    else:
        # Inverted logics to avoid replacements of @http-like mentions
        text = re.sub(r"@[A-Za-z0-9_-]+", "[USER]", text) # @user on Twitter and Gab only
        text = re.sub(r"http[^\r\n\t\f\v )\]\}]+", "[URL]", text) # excludes trailing parentheses too

    # Replace emojis (if specified)
    if keep_emojis is False:
        text = convert_emojis_to_text(text)

    # Segment hashtags, and clean newlines and tabs
    text = re.sub(r"#[A-Za-z0-9]+", regex_match_segmentation, text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("[linebreak]", " ") # newlines as in Cad data

    # @TODO: Split names of Reddit channels?

    # Replace non-standard characters with simple space
    text = text.replace(u'\xa0', u' ')    # no-break space
    text = text.replace(u'\u200d', u' ')  # zero-width joiner
    
    return text


def preprocess_dataset(input_filepath, output_folder, dataset_name, keep_emojis):

    def get_label_truth(instance, labels):
        labels_count = Counter(labels)

        # If there is no single truth mark it for subsequent removal
        # Note: in CAD a post can be multi-label, so it is not removed here (instead, we keep multiple "$"-separated labels)
        if len(labels_count) == 1:
            return list(labels_count.keys())[0]
        else:
            return AMBIGUOUS_LABEL_NAME
    
    raw_instances, instances = dict(), dict()
    lines_counter, duplicates_counter, extra_counter, ambiguous_counter = 0, 0, 0, 0

    # Create a dictionary from the input excluding extra labels and keeping track of duplicates
    with open(input_filepath, "r") as f:
        for line in f:
            # Store the main attributes of the example
            raw_label, raw_text = line.rstrip().split("\t")

            # Perform data cleaning and anonymization for the text
            text = clean_text(raw_text, dataset_name, keep_emojis)
            
            # Discard the example if it belongs to a predefined extra label
            if raw_label in EXTRA_LABELS[dataset_name]:
                extra_counter += 1
            # O.w., add it to an instance dictionary (possibly attaching the new label, if duplicate)
            else:
                if text not in raw_instances.keys():
                    raw_instances[text] = [raw_label]
                else:
                    raw_instances[text].append(raw_label)
                    duplicates_counter += 1
            lines_counter += 1

    # Evaluate label truth for duplicates and discard ambiguous instances
    for text, labels in raw_instances.items():
        label = get_label_truth(text, labels)
        if label == AMBIGUOUS_LABEL_NAME:
            ambiguous_counter += 1
        else:
            instances[text] = label

    # Log the main statistics about data preprocessing/cleaning
    logging.info(f"{duplicates_counter} duplicates, {extra_counter} extra, {ambiguous_counter} ambiguous instances have been removed.")
    logging.info(f"Resulting dataset size: {len(instances)} (out of {lines_counter} original instances).\n")

    # Write preprocessed/cleaned dataset to the output file in the format "label[\t]text"
    with open(os.path.join(output_folder, dataset_name + ".all"), "w") as f:
        for text, label in instances.items():
            f.write(f"{label}\t{text}\n")


def split_dataset(output_folder, dataset_name, k=10):
    id_to_text, data_splits = dict(), dict()
    labels = []

    # Filepath to the preprocessed dataset to shuffle and split
    input_filepath = os.path.join(output_folder, dataset_name + ".all")

    # Shuffle dataset prior to splitting
    with open(input_filepath, "r") as f:
        lines = f.readlines()
    random.seed(RANDOM_SEED)
    random.shuffle(lines)
    os.remove(input_filepath)
    with open(input_filepath, "w") as f:
        f.writelines(lines)

    # Store instance IDs, texts, and labels from the input file
    instance_id = 0
    with open(input_filepath, "r") as f:
        for line in f:
            label, text = line.rstrip().split("\t")
            id_to_text[instance_id] = text
            labels.append(LABEL_TO_ID[dataset_name][label])
            instance_id += 1

    # Create numpy arrays and stratified k-fold cross-validation object
    X, y = np.zeros(len(labels)), np.array(labels)
    cv_object = StratifiedKFold(n_splits=k) # shuffle=True, random_state=42

    split_id = 0
    lines_counter, hateful_counter, abusive_counter = 0, 0, 0
    for _, split_indices in cv_object.split(X, y):
        # Split data according to labels and log descriptive statistics
        split_labels = y[split_indices].tolist()
        logging.info(f"[Split #{split_id}] Instances: {len(split_indices)}")
        for label_id in range(len(LABEL_TO_ID[dataset_name].values())):
            logging.info(f"  {label_id} ({ID_TO_LABEL[dataset_name][label_id]}): {split_labels.count(label_id)}")

        # Determine the train/dev/test split in which to write the current data portion
        if split_id in list(range(k-2)):
            file_ext = "train"
        elif split_id == (k-2):
            file_ext = "dev"
        elif split_id == (k-1):
            file_ext = "test"
        else:
            sys.exit(f"Split #{split_id} does not exist.")

        # Write the current data portion in its original multiclass form
        output_mc_filepath = os.path.join(output_folder, dataset_name + "-mc" + "." + file_ext)
        f = open(output_mc_filepath, "a")
        for i in range(len(split_indices)):
            f.write(str(split_labels[i]) + "\t" + id_to_text[split_indices[i]] + "\n")
        f.close()

        # Write the current data portion in the binarized (hateful, non-hateful) form
        output_bn_filepath = os.path.join(output_folder, dataset_name + "." + file_ext)
        f = open(output_bn_filepath, "a")
        for i in range(len(split_indices)):
            binarized_label = LABEL_MAPPING[dataset_name][split_labels[i]]
            f.write(str(binarized_label) + "\t" + id_to_text[split_indices[i]] + "\n")
            if binarized_label == 1: hateful_counter += 1
            if split_labels[i] in ABUSIVE_IDS[dataset_name]: abusive_counter += 1
            lines_counter += 1
        f.close()

        split_id += 1

    # Shuffle each split after partitioning (some classifiers do not do that prior to training)
    for ext in ["train", "dev", "test"]:
        for variant in ["", "-mc"]:
            filepath = os.path.join(output_folder, dataset_name + variant + "." + ext)
            with open(filepath, "r") as f:
                lines = f.readlines()
            random.seed(RANDOM_SEED)
            random.shuffle(lines)
            os.remove(filepath)
            with open(filepath, "w") as f:
                f.writelines(lines)

    # Log descriptive statistics about the binarized dataset
    hateful_density = (hateful_counter / lines_counter) * 100
    abusive_density = (abusive_counter / lines_counter) * 100

    # Stormfront has hateful and abusive classes which match
    if dataset_name == "stormfront":
        toxic_density = hateful_density
    else:
        toxic_density = ((hateful_counter + abusive_counter) / lines_counter) * 100

    logging.info(f"\nHateful: {hateful_counter}; Non-hateful: {lines_counter-hateful_counter}. Hateful density: {hateful_density:.2f}%.")
    logging.info(f"Abusive: {abusive_counter}; Non-abusive: {lines_counter-abusive_counter if lines_counter>0 else 0}. Abusive density: {abusive_density:.2f}%.")
    logging.info(f"Toxic: {hateful_counter+abusive_counter}; Non-toxic: {lines_counter-(hateful_counter+abusive_counter)}. Toxic density: {toxic_density:.2f}%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input_folder", type=str, required=True, 
        help="The path to the raw input folder.")
    parser.add_argument("-O", "--output_folder", type=str, required=False, 
        default="data", help="The path where to store the preprocessed train/dev/test splits.")
    parser.add_argument("-N", "--dataset_name", type=str, required=True, 
        choices=["founta", "gab", "stormfront", "cad"], 
        help="The name of the dataset, used for naming the splits.")
    parser.add_argument("-E", "--keep_emojis", default=True, action="store_false", 
        help="Whether or not keeping the raw emojis in the texts.")
    parser.add_argument("-V", "--verbose", default=False, action="store_true", 
        help="Whether or not showing all debug information about the execution.")
    args = parser.parse_args()

    # Set the logger (folder, filepath, verbosity)
    if not os.path.exists("logs"): os.makedirs("logs")
    logger_filename = os.path.join("logs", "prepare_data_" + args.dataset_name + ".log")
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, filename=logger_filename, 
            filemode="w", format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, filename=logger_filename, 
            filemode="w", format="%(message)s")

    # Create a clean two-column version of the dataset, regardless of the original format
    create_dataset_file(args.input_folder, args.dataset_name)
    args.input_filepath = os.path.join(args.input_folder, "data.tsv")

    # Preprocess and return dataset instances
    logging.info(f"PREPROCESSING {args.dataset_name}...")
    preprocess_dataset(args.input_filepath, args.output_folder, args.dataset_name, args.keep_emojis)

    # Partition the preprocessed dataset into train/dev/test splits
    logging.info(f"SPLITTING {args.dataset_name}...")
    split_dataset(args.output_folder, args.dataset_name, k=10)
