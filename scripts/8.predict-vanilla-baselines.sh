declare -a PRETRAINED_MODELS=("bert")
declare -a DATASETS=("founta" "gab" "stormfront" "cad")
declare -a SEED_IDS=("1" "2" "3")

RESULTS_FOLDER="results"
BERT_RESULTS_FOLDER="results/bert"
TEST_DATA_FOLDER="machamp/data"
METHOD_NAME="vanilla"

mkdir $RESULTS_FOLDER
mkdir $BERT_RESULTS_FOLDER

for SOURCE_DATASET in "${DATASETS[@]}"
do
    mkdir $BERT_RESULTS_FOLDER/$SOURCE_DATASET
done


for PRETRAINED_MODEL in "${PRETRAINED_MODELS[@]}"
do
    for SOURCE_DATASET in "${DATASETS[@]}"
    do
        for TARGET_DATASET in "${DATASETS[@]}"
        do
            for SEED_ID in "${SEED_IDS[@]}"
            do
                MODEL_NAME=$PRETRAINED_MODEL.$SOURCE_DATASET.$METHOD_NAME.$SEED_ID
                MODEL_FILEPATH=$(ls -td logs/$MODEL_NAME/*/ | head -1)

                echo "Testing '$MODEL_NAME' on '$TARGET_DATASET'..."
                python machamp/predict.py \
                    $MODEL_FILEPATH/model.tar.gz \
                    $TEST_DATA_FOLDER/$TARGET_DATASET.test \
                    $BERT_RESULTS_FOLDER/$SOURCE_DATASET/$MODEL_NAME.$TARGET_DATASET.out \
                    --device 0
                echo "==> Done."
            done

            CURR_BASE_MODEL_NAME=$PRETRAINED_MODEL.$SOURCE_DATASET.$METHOD_NAME
            CURR_BASE_FILEPATH=$BERT_RESULTS_FOLDER/$SOURCE_DATASET/$CURR_BASE_MODEL_NAME
            python scripts/compute_avg_score_acc.py -I $CURR_BASE_FILEPATH -T $TARGET_DATASET -N 3
        done
    done
done