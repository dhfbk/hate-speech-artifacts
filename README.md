# Features or Spurious Artifacts? Data-centric Baselines for Fair and Robust Hate Speech Detection

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Based on MaChAmp v0.2](https://img.shields.io/badge/MaChAmp-v0.2-blue)](https://github.com/machamp-nlp/machamp)

This repository contains code and resources associated to the paper:

Alan Ramponi and Sara Tonelli. 2022. **Features or Spurious Artifacts? Data-centric Baselines for Fair and Robust Hate Speech Detection**. In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 3027–3040, Seattle, United States. Association for Computational Linguistics. [[cite]](#citation) [[paper]](https://aclanthology.org/2022.naacl-main.221/)

**Useful resources**

- :memo: **Lexical artifacts statement template** ([here](artifacts-statement/README.md))
- :pencil2: **Annotated lexical artifacts** ([here](resources/artifacts/README.md))
- :gear: **Fine-tuned language models** ([here](resources/models/README.md))
- :rocket: **`lexartifacts` package to ease documentation** ([here](lexartifacts-package/README.md))


## Getting started

To get started, clone this repository on your own path:
```
git clone https://github.com/dhfbk/hate-speech-artifacts.git
```


### Environment

Create an environment with your own preferred package manager. We used [python 3.8](https://www.python.org/downloads/release/python-380/) and dependencies listed in [`requirements.txt`](requirements.txt). If you use [conda](https://docs.conda.io/en/latest/), you can just run the following commands from the root of the project:

```
conda create --name lex-artifacts python=3.8      # create the environment
conda activate lex-artifacts                      # activate the environment
pip install --user -r requirements.txt            # install the required packages
```


### Data

Get the datasets we used in our experiments following the instructions below.

- **Reddit** (CAD v1.1; Vidgen et al., 2021): publicly available [here](http://doi.org/10.5281/zenodo.4881008)
  - Download data, then put the `cad_v1_1.tsv` file into `data/raw/cad/`
- **Twitter** (Founta et al., 2018): available upon formal request to data owners [here](https://zenodo.org/record/3706866#.Ylq9fZNBzyi)
  - Download data, then put the `hatespeech_text_label_vote_RESTRICTED_100K.csv` file into `data/raw/founta/`
- **Gab** (GHC v2021-03-03; Kennedy et al., 2021): publicly available [here](https://osf.io/edua3/)
  - Download data, then put `GabHateCorpus_annotations.tsv`, `ghc_train.tsv` and ` ghc_test.tsv` files into `data/raw/gab/`
- **Stormfront** (de Gibert et al., 2018): publicly available [here](https://github.com/Vicomtech/hate-speech-dataset)
  - Download data, then put the `all_files/` folder and the `annotations_metadata.csv` file into `data/raw/stormfront/`


## Running the experiments


### 1. Data preparation

Prepare data by preprocessing, cleaning and splitting all raw datasets, also getting relevant statistics:
```
sh scripts/1.prepare-data.sh
```

Splits will be created on the `machamp/data/` folder, with name `$DATASET_NAME.$SPLIT`, where:
- `$DATASET_NAME`: the name of the dataset (`cad`, `founta`, `gab`, or `stormfront`)
- `$SPLIT`: the data split portion (`train`, `dev`, or `test`)

Extras: Logs for this step, including relevant statistics, will be written in `logs/prepare_data_$DATASET_NAME.log`. Further, intermediate files can be found on the `data/` folder, i.e., data before splitting (`$DATASET_NAME.all`) as well as before binarizing labels (`$DATASET_NAME-mc.$SPLIT`).


### 2. Training the baselines


#### BERT baselines

Fine-tune the vanilla BERT baselines on all datasets, using multiple seeds:
```
sh scripts/2.train-vanilla-baselines.sh
```

Models will be on the `logs/bert.$DATASET_NAME.vanilla.$SEED_ID/$DATETIME/` folders, where:
- `$DATASET_NAME`: the name of the dataset (`cad`, `founta`, `gab`, or `stormfront`)
- `$SEED_ID`: an incremental integer corresponding to a seed (`1`, `2`, or `3`)
- `$DATETIME`: the exact datetime corresponding to when you ran the fine-tuning process

Within these folders, you can find fine-tuned models (`model.tar.gz`) to be used for inference. Additionally, you can find training dynamics files on the `training_dynamics/` folder, which are necessary if you need to filter data and thus run the FILTERING approach described in the following.


#### Filtering baselines

Fine-tune the filtered BERT baselines on all datasets, using multiple seeds:
```
sh scripts/2.train-filtering-baselines.sh
```

Models will be on the `logs/bert.$DATASET_NAME.f$THRESHOLD.$SEED_ID/$DATETIME/` folders, where:
- `$DATASET_NAME`: the name of the dataset (`cad`, `founta`, `gab`, or `stormfront`)
- `$SEED_ID`: an incremental integer corresponding to a seed (`1`, `2`, or `3`)
- `$THRESHOLD`: the percentage of most ambiguous training data points to be used (`25`, `33`, or `50`)
- `$DATETIME`: the exact datetime corresponding to when you ran the fine-tuning process

Within these folders, you can find fine-tuned models (`model.tar.gz`) to be used for inference.

Note: to run these baselines you have to filter training data based on training dynamics you can find on the `training_dynamics/` folder by using the [dataset cartography official code](https://github.com/allenai/cartography).


### 3. Compute per-corpus lexical artifacts

Compute the contribution strength of each token to each label, divided by dataset:
```
sh scripts/3.compute-contributions.sh
```

You can then find the results on `results/token-contributions/bert-base-uncased/$DATASET_NAME.all/`, where:
- `$DATASET_NAME`: the name of the dataset (`cad`, `founta`, `gab`, or `stormfront`)

The files of interest within these folders are named `pmi.all-vs-hate.hate.tsv` (tab-separated file with tokens and scores), i.e., the contribution strength of each token for the hateful class w.r.t. the rest (i.e., non-hateful).


### 4. Compute cross-corpora lexical artifacts

Compute the cross-corpora contribution strength of each token to each label:
```
sh scripts/4.identify-artifacts.sh
```

You will find the results on `results/token-contributions/bert-base-uncased/summary/`. The files of interest within these folders are named `pmi.all-vs-hate.hate.tsv` (tab-separated file with tokens and scores for each platform, along with their mean and std), i.e., the contribution strength of each token for the hateful class w.r.t. the rest (i.e., non-hateful examples).


### 5. Create masking and removal variants

Create `masking` and `removal` variants of train/dev data for all corpora given annotated lexical artifacts types (i.e., [spurious identity-related artifacts](resources/artifacts/sp-id.txt) and [spurious non identity-related artifacts](resources/artifacts/sp-nid.txt):

```
sh scripts/5.create-variants.sh
```

You will find train/dev on `machamp/data/$DATASET_NAME.bert.$APPROACH-$ARTIFACTS_TYPE.$SPLIT_NAME`, where:
- `$DATASET_NAME`: the name of the dataset (`cad`, `founta`, `gab`, or `stormfront`)
- `$APPROACH`: the approach for lexical artifacts, i.e. `del` (removal) or `mask` (masking)
- `$ARTIFACTS_TYPE`: the considered lexical artifacts (`sp-id` (identity) or `sp-nid` (non-identity))
- `$SPLIT_NAME`: the data partition (`train` or `dev`)


### 6. Create identity-related test subsets

Create subsets of test data containing spurious identity-related artifacts to test unintended bias towards identities:

```
sh scripts/6.create-subsets.sh
```

You will find the test files on `machamp/data/$DATASET_NAME.bert.sp-i.test`, where:
- `$DATASET_NAME`: the name of the dataset (`cad`, `founta`, `gab`, or `stormfront`)


### 7. Training masking and removal variants

Fine-tune spurious artifacts-aware variants (identity- and non identity-related) on all corpora w/ multiple seeds:

```
sh scripts/7.train-sp-id-nid-methods.sh
```

Models will be on `logs/bert.$DATASET_NAME.$APPROACH-$ARTIFACTS_TYPE.$SEED_ID/$DATETIME/` folders, where:
- `$DATASET_NAME`: the name of the dataset (`cad`, `founta`, `gab`, or `stormfront`)
- `$APPROACH`: the approach for lexical artifacts, i.e. `del` (removal) or `mask` (masking)
- `$ARTIFACTS_TYPE`: the considered lexical artifacts (`sp-id` (identity) or `sp-nid` (non-identity))
- `$SEED_ID`: an incremental integer corresponding to a seed (`1`, `2`, or `3`)
- `$DATETIME`: the exact datetime corresponding to when you ran the fine-tuning process

Within these folders, you can find fine-tuned models (`model.tar.gz`) to be used for inference.


### 8. Inference and evaluation

#### Performance on full test sets

Predict and evaluate vanilla, filtering, and artifacts-aware methods on full data, according to the macro F1 score:

```
# Inference and evaluation for vanilla baselines
sh scripts/8.predict-vanilla-baselines.sh

# Inference and evaluation for filtering baselines
sh scripts/8.predict-filtering-baselines.sh

# Inference and evaluation for artifacts-aware methods
sh scripts/8.predict-mask-del-methods.sh
```

#### Performance on subsets of test data containing spurious identity-related artifacts

Predict and evaluate vanilla, filtering, and artifacts-aware methods on subsets of test data containing spurious identity-related artifacts to test unintended bias towards identities, according to the FPR score:

```
# Inference and evaluation for vanilla baselines
sh scripts/8.predict-vanilla-baselines-subset.sh

# Inference and evaluation for filtering baselines
sh scripts/8.predict-filtering-baselines-subset.sh

# Inference and evaluation for artifacts-aware methods
sh scripts/8.predict-mask-del-methods-subset.sh
```


## Citation

If you use or build on top of this work, please cite our paper as follows:

```
@inproceedings{ramponi-tonelli-2022-features,
    title = "Features or Spurious Artifacts? Data-centric Baselines for Fair and Robust Hate Speech Detection",
    author = "Ramponi, Alan  and
      Tonelli, Sara",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.221",
    pages = "3027--3040",
}
```