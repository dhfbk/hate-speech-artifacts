# :pencil2: Annotated lexical artifacts

This page presents the annotated lexical artifacts described in the paper.

The top lexical artifacts across all platforms are available in a tab-separated format ([here](annotated_artifacts.tsv)). We provide both in-platform and cross-platform scores, as well as disaggregated annotations. Specifically, the lexical artifacts file consists of the following columns:

- `token`: the token given by the BERT tokenizer
- `founta`: the in-platform PMI score for the token (Twitter)
- `gab`: the in-platform PMI score for the token (Gab)
- `stormfront`: the in-platform PMI score for the token (Stormfront)
- `cad`: the in-platform PMI score for the token (Reddit)
- `mean`: the average cross-platform PMI score for the token
- `std`: the stddev cross-platform PMI score for the token
- `a1-hateful`: whether the token *potentially* conveys hatefulness, profanity, or is otherwise frequently associated with hateful contexts (annotator 1)
- `a1-identity`: whether the token *potentially* refers to minority identities (annotator 1)
- `a2-hateful`: whether the token *potentially* conveys hatefulness, profanity, or is otherwise frequently associated with hateful contexts (annotator 2)
- `a2-identity`: whether the token *potentially* refers to minority identities (annotator 2)

We also provide the lists of [spurious identity-related artifacts](sp-id.txt) and [spurious non identity-related artifacts](sp-nid.txt) that do not exhibit disagreement after adjudication.