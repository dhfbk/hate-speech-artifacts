declare -a PRETRAINED_MODELS=("bert")
declare -a DATASETS=("founta" "gab" "stormfront" "cad")
declare -a APPROACHES=("mask" "del")
declare -a ARTIFACTS_TYPES=("sp-id" "sp-nid")
declare -a SEED_IDS=("1" "2" "3")

TEST_DATA_FOLDER="machamp/data"


for PRETRAINED_MODEL in "${PRETRAINED_MODELS[@]}"
do
    MODEL_RESULTS_FOLDER=results/$PRETRAINED_MODEL

    for SOURCE_DATASET in "${DATASETS[@]}"
    do
        for APPROACH in "${APPROACHES[@]}"
        do
            for ARTIFACTS_TYPE in "${ARTIFACTS_TYPES[@]}"
            do
                for TARGET_DATASET in "${DATASETS[@]}"
                do
                    for SEED_ID in "${SEED_IDS[@]}"
                    do
                        MODEL_NAME=$PRETRAINED_MODEL.$SOURCE_DATASET.$APPROACH-$ARTIFACTS_TYPE.$SEED_ID
                        MODEL_FILEPATH=$(ls -td logs/$MODEL_NAME/*/ | head -1)

                        echo "Testing '$MODEL_NAME' on '$TARGET_DATASET'..."
                        python machamp/predict.py \
                            $MODEL_FILEPATH/model.tar.gz \
                            $TEST_DATA_FOLDER/$TARGET_DATASET.$PRETRAINED_MODEL.sp-i.test \
                            $MODEL_RESULTS_FOLDER/$SOURCE_DATASET/$MODEL_NAME.$TARGET_DATASET.subset.out \
                            --device 0

                        python scripts/compute_fpr.py \
                            --input_pred_filepath $MODEL_RESULTS_FOLDER/$SOURCE_DATASET/$MODEL_NAME.$TARGET_DATASET.subset.out \
                            --input_gold_filepath $TEST_DATA_FOLDER/$TARGET_DATASET.$PRETRAINED_MODEL.sp-i.test \
                            --output_filepath $MODEL_RESULTS_FOLDER/$SOURCE_DATASET/$MODEL_NAME.$TARGET_DATASET.fairness

                        echo "==> Done."
                    done

                    CURR_BASE_MODEL_NAME=$PRETRAINED_MODEL.$SOURCE_DATASET.$APPROACH-$ARTIFACTS_TYPE
                    CURR_BASE_FILEPATH=$MODEL_RESULTS_FOLDER/$SOURCE_DATASET/$CURR_BASE_MODEL_NAME
                    python scripts/compute_avg_score_fpr.py -I $CURR_BASE_FILEPATH -T $TARGET_DATASET -N 3
                done
            done
        done
    done
done
