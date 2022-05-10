# Preprocessing, cleaning, and splitting of raw datasets, with copy to the classifier data folder.

# Variables declaration
declare -a DATASETS=("founta" "gab" "stormfront" "cad")
declare -a SPLIT_NAMES=("train" "dev" "test")
INPUT_DATA_FOLDER="data/raw"
OUTPUT_DATA_FOLDER="data"
CLASSIFIER_DATA_FOLDER="machamp/data"


# Preprocess, clean, and split all datasets
for DATASET in "${DATASETS[@]}"
do
    echo "Preprocessing '$DATASET' dataset..."
    python scripts/1.prepare-data.py -I $INPUT_DATA_FOLDER/$DATASET -O $OUTPUT_DATA_FOLDER -N $DATASET
    echo "==> Done."
done

# Move data splits to the classifier data folder for the experiments
for DATASET in "${DATASETS[@]}"
do
    for SPLIT_NAME in "${SPLIT_NAMES[@]}"
    do
        echo "Copying '$DATASET' $SPLIT_NAME split to the classifier data folder..."
        mkdir -p $CLASSIFIER_DATA_FOLDER
        cp $OUTPUT_DATA_FOLDER/$DATASET.$SPLIT_NAME $CLASSIFIER_DATA_FOLDER
        echo "==> Done."
    done
done