# Creating masking and removal variants of train/dev data for all corpora given annotated lexical artifacts types.

declare -a DATASETS=("founta" "gab" "stormfront" "cad")
declare -a SPLIT_NAMES=("train" "dev")
declare -a PRETRAINED_MODELS=("bert")
declare -a ARTIFACTS_TYPES=("sp-id" "sp-nid")
declare -a APPROACHES=("mask" "del")

CONTRIBUTIONS_DATA_FOLDER="results/token-contributions"
ARTIFACTS_FOLDER="artifacts"


for DATASET in "${DATASETS[@]}"
do
    for SPLIT_NAME in "${SPLIT_NAMES[@]}"
    do
        echo "Create variants for 'machamp/data/$DATASET.$SPLIT_NAME' file..."
        for PRETRAINED_MODEL in "${PRETRAINED_MODELS[@]}"
        do
            for ARTIFACTS_TYPE in "${ARTIFACTS_TYPES[@]}"
            do
                for APPROACH in "${APPROACHES[@]}"
                do
                    if [[ $PRETRAINED_MODEL == "bert" ]]
                    then
                        PRETRAINED_MODEL_FULLNAME="bert-base-uncased"
                    else
                        echo "'$PRETRAINED_MODEL' is not supported. Exiting."
                        exit 1
                    fi
                    ARTIFACTS_FILEPATH=$CONTRIBUTIONS_DATA_FOLDER/$PRETRAINED_MODEL_FULLNAME/$ARTIFACTS_FOLDER/$ARTIFACTS_TYPE.txt

                    python scripts/5.create-variants.py \
                        --input_filepath machamp/data/$DATASET.$SPLIT_NAME \
                        --output_filepath machamp/data/$DATASET.$PRETRAINED_MODEL.$APPROACH-$ARTIFACTS_TYPE.$SPLIT_NAME \
                        --artifacts_filepath $ARTIFACTS_FILEPATH \
                        --strategy $APPROACH \
                        --classes all \
                        --pretrained_tokenizer $PRETRAINED_MODEL_FULLNAME
                done
            done
        done
        echo "==> Done."
    done
done
