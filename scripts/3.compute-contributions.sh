# Computing the contribution strengths of intra-corpora tokens according to label divisions
# (here, so it can be used afterwards either for model-based debiasing or cross-corpora analyses).

# Variables declaration
declare -a PRETRAINED_MODELS=("bert-base-uncased")
declare -a DATASETS=("founta" "gab" "stormfront" "cad")
declare -a PARTITIONS=("all-vs-hate")


# For each model, dataset and partition, compute and write token contribution strengths
for MODEL in "${PRETRAINED_MODELS[@]}"
do
	for DATASET in "${DATASETS[@]}"
	do
		for PARTITION in "${PARTITIONS[@]}"
		do
		    echo "Compute '$PARTITION' token contributions for the '$DATASET' dataset..."
			python scripts/3.compute-contributions.py -D $DATASET -L $PARTITION -T $MODEL
			echo "==> Done."
		done
	done
done