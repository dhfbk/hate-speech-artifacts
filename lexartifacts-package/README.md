# :rocket: `lexartifacts` python package

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![v0.1.0](https://img.shields.io/badge/pypi-v0.1.0-orange)](https://pypi.org/project/lexartifacts/0.1.0/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue)](#)

**`lexartifacts`** is a python package for **easy computation and documentation of lexical artifacts** in text classification datasets. It was firstly introduced in [1] in order to make the process of documenting lexical artifacts as smooth as possible – and thus to allow widespread adoption of artifacts statement in the future. 

With just **2 lines of code** you can compute lexical artifacts on your own dataset and automatically generate outputs in different formats (from a simple raw `.tsv` table, to complete `.txt` or `tex` lexical artifacts statement) for **seamless inclusion in publications**.


> [1] Alan Ramponi and Sara Tonelli. 2022. **Features or Spurious Artifacts? Data-centric Baselines for Fair and Robust Hate Speech Detection**. In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. [[cite]](#citation) [pdf (coming soon)]

:warning: **Warning**. *The package is a **beta release** that has been thoroughly tested on English and BERT-like tokenizers only. We are extending it to multiple languages and to all non-BERT-like tokenizers: stay tuned!* :rocket:


## Installation

The package can be installed using [`pip`](https://pypi.org/project/lexartifacts/) as follows:

```
pip install lexartifacts
```

## Usage

After importing the package and preparing your own data in two lists (`texts` and associated `labels`), there are just two functions you will need to use:
- **`compute`**: computes lexical artifacts given `texts`, `labels` and a `label_of_interest`, returning a dataframe `artifacts_df` with lexical artifacts sorted by score;
- **`make_report`**: generates a report of `top_k` lexical artifacts in a format `output_format` from the previously created dataframe `artifacts_df`.

Below there is a minimal example. Please check [`compute`](#compute-documentation) and [`make_report`](#make_report-documentation) for a full documentation.


### Minimal example

Compute and generate a report of the top-*20* *abusive* lexical artifacts in a *tex* format:

```python
from lexartifacts import lexical_artifacts

texts = ["you are a fucking post!", "fucking post should suffer", "it's fucking great!", ...]
labels = ["abusive", "abusive", "not-abusive", ...]

# Compute lexical artifacts for the dataset with focus on label "abusive"
artifacts_df = lexical_artifacts.compute(texts=texts, labels=labels, label_of_interest="abusive")

# Generate report for the artifacts dataframe in the desired format
lexical_artifacts.make_report(artifacts_df=artifacts_df, top_k=20, output_format="tex")
```

The output will be printed on the console:

```latex
\textsc{i) Top lexical artifacts.}~~~We present the top $k=20$ most informative tokens for the \emph{abusive} class along with their scores in Table \ref{tab:top-k-lexical-artifacts}.

\begin{table}
    \centering
    \begin{tabular}{rlr}
        \toprule
        \emph{Rank} & Token & Score \\
        \midrule
        1 & post & 1.0 \\
        2 & kill & 0.85 \\
        3 & fucking & 0.77 \\
        4 & fake & 0.62 \\
        % ...
        20 & suffer & 0.23 \\
        \bottomrule
    \end{tabular}
    \caption{\label{tab:top-k-lexical-artifacts} Top 20 most informative tokens for the abusive class according to PMI.}
\end{table}

\textsc{ii) Class definitions.}~~~[INSERT HERE THE DEFINITION FOR THE \emph{abusive} CLASS.]

\textsc{iii) Methods and resources.}~~~In order to compute the correlation between tokens to the \emph{abusive} class we employ [INSERT HERE THE METHOD USED FOR COMPUTING LEXICAL ARTIFACTS].
[INSERT HERE THE DETAILS ON PREPROCESSING AND DUPLICATES HANDLING.]
[INSERT HERE THE LINKS TO RELATED RESOURCES (e.g., FULL LIST OF LEXICAL ARTIFACTS).]

% Notes: for correct table formatting, include the following:
% \usepackage{{booktabs}}
```

### Documentation

Full documentation for the `compute` and `make_report` functions are provided below.

#### `compute` documentation

```
Parameters
----------
texts: List[str]
    Input texts (note: the ith text of "texts" must match the ith label of "labels")
labels: List[str]
    Input labels (note: the ith label of "labels" must match the ith text of "texts")
label_of_interest: str
    Label that is the focus of the artifacts calculation (note: it must be in "labels")
method: str
    Algorithm to compute the contribution strength of each token to each label. Default: "pmi"
    For now, we support "pmi" as implemented in [1], more on next releases
special_tokens: List[str]
    List of special tokens to add to the tokenizer's vocabulary. Default: []
add_emojis: bool
    Whether or not adding emojis to the tokenizer's vocabulary. Default: True
    If this is set to False, a special token "[EMOJI]" will be used for all emojis
stopwords: str
    The language for the stopwords to be removed from lexical artifacts. Default: en (English)
    If None, all stopwords are instead retained in the list of lexical artifacts
    For now, only "en" is supported (with a default stopword list), more on next releases
pretrained_tokenizer: str
    Name of the HuggingFace's pretrained tokenizer to use (e.g., "bert-base-uncased")
    For now, BPE-based tokenizers (e.g., RoBERTa-base, GPT2) would not filter stopword 
    correctly due to the "Ġ" special character. Thorough support on next releases

Returns
-------
sorted_pmi_scores: pd.core.frame.DataFrame
    Pandas dataframe with tokens as rows and label_of_interest as column. Values in this matrix 
    are PMI scores following the implementation by [1].
```

#### `make_report` documentation

```
Parameters
----------
artifacts_df: pd.core.frame.DataFrame
    Pandas dataframe with tokens as rows and label_of_interest as column
top_k: int
    Number of top artifacts to list. Default: 20
output_format: str
    Format for the lexical artifacts output (choices: tsv, txt, tex). Default: tsv
output_filepath: str
    Optional filepath where to write the output. Default: None (i.e., console print)
```


## Citation

If you use or build on top of the `lexartifacts` python package, please cite the work as follows:

```
@inproceedings{ramponi-tonelli-2022-features,
    title = "Features or Spurious Artifacts? Data-centric Baselines for Fair and Robust Hate Speech Detection",
    author = "Ramponi, Alan and Tonelli, Sara",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, Washington, USA"
}
```