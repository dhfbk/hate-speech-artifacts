# Training of masking/removal methods of spurious artifacts (both identity- and non identity-related).

# Variables declaration
declare -a PRETRAINED_MODELS=("bert")
declare -a DATASETS=("founta" "gab" "stormfront" "cad")
declare -a APPROACHES=("mask" "del")
declare -a ARTIFACTS_TYPES=("sp-id" "sp-nid")
declare -a SEED_IDS=("1" "2" "3")


# Train baselines using default hyperparameters from previous work
for MODEL in "${PRETRAINED_MODELS[@]}"
do
    for DATASET in "${DATASETS[@]}"
    do
        for APPROACH in "${APPROACHES[@]}"
        do
            for ARTIFACTS_TYPE in "${ARTIFACTS_TYPES[@]}"
            do
                for SEED_ID in "${SEED_IDS[@]}"
                do
                    python machamp/train.py \
                        --dataset_config machamp/configs/$DATASET.$MODEL.$APPROACH-$ARTIFACTS_TYPE.json \
                        --parameters_config machamp/configs/params.$MODEL.$SEED_ID.json \
                        --name $MODEL.$DATASET.$APPROACH-$ARTIFACTS_TYPE.$SEED_ID \
                        --device -0
                done
            done
        done
    done
done
