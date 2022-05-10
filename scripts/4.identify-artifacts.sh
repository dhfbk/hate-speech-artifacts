# Computing the contribution strengths of inter-corpora tokens according to label divisions
# (here, because it requires a preliminar computation of intra-corpora token contributions).

# Variables declaration
declare -a PRETRAINED_MODELS=("bert-base-uncased")
declare -a PARTITIONS=("all-vs-hate")


# For each model and partition, compute and write token contribution strengths
for MODEL in "${PRETRAINED_MODELS[@]}"
do
	for PARTITION in "${PARTITIONS[@]}"
	do
	    echo "Compute '$PARTITION' cross-corpora token contributions..."
		python scripts/4.identify-artifacts.py -L $PARTITION -T $MODEL
		echo "==> Done."
	done
done