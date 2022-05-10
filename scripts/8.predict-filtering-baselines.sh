declare -a PRETRAINED_MODELS=("bert")
declare -a DATASETS=("founta" "gab" "stormfront" "cad")
declare -a THRESHOLDS=("f25" "f33" "f50")
declare -a SEED_IDS=("1" "2" "3")

TEST_DATA_FOLDER="machamp/data"


for PRETRAINED_MODEL in "${PRETRAINED_MODELS[@]}"
do
    MODEL_RESULTS_FOLDER=results/$PRETRAINED_MODEL

    for SOURCE_DATASET in "${DATASETS[@]}"
    do
        for THRESHOLD in "${THRESHOLDS[@]}"
        do
            for TARGET_DATASET in "${DATASETS[@]}"
            do
                for SEED_ID in "${SEED_IDS[@]}"
                do
                    MODEL_NAME=$PRETRAINED_MODEL.$SOURCE_DATASET.$THRESHOLD.$SEED_ID
                    MODEL_FILEPATH=$(ls -td logs/$MODEL_NAME/*/ | head -1)

                    echo "Testing '$MODEL_NAME' on '$TARGET_DATASET'..."
                    python machamp/predict.py \
                        $MODEL_FILEPATH/model.tar.gz \
                        $TEST_DATA_FOLDER/$TARGET_DATASET.test \
                        $MODEL_RESULTS_FOLDER/$SOURCE_DATASET/$MODEL_NAME.$TARGET_DATASET.out \
                        --device 0
                    echo "==> Done."
                done

                CURR_BASE_MODEL_NAME=$PRETRAINED_MODEL.$SOURCE_DATASET.$THRESHOLD
                CURR_BASE_FILEPATH=$MODEL_RESULTS_FOLDER/$SOURCE_DATASET/$CURR_BASE_MODEL_NAME
                python scripts/compute_avg_score_acc.py -I $CURR_BASE_FILEPATH -T $TARGET_DATASET -N 3
            done
        done
    done
done
