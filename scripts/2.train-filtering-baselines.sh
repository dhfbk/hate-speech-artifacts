# Training of filtering baselines (i.e., with data filtered according to training dynamics of corresponding baselines)
# Note: training data files can be created using the data maps codebase: https://github.com/allenai/cartography

# Variables declaration
declare -a PRETRAINED_MODELS=("bert")
declare -a DATASETS=("founta" "gab" "stormfront" "cad")
declare -a THRESHOLDS=("f25" "f33" "f50")
declare -a SEED_IDS=("1" "2" "3")

# Train baselines using default hyperparameters from previous work
for MODEL in "${PRETRAINED_MODELS[@]}"
do
    for DATASET in "${DATASETS[@]}"
    do
        for THRESHOLD in "${THRESHOLDS[@]}"
        do
            for SEED_ID in "${SEED_IDS[@]}"
            do
                python machamp/train.py \
                    --dataset_config machamp/configs/$DATASET.$MODEL.$THRESHOLD.$SEED_ID.json \
                    --parameters_config machamp/configs/params.$MODEL.1.json \
                    --name $MODEL.$DATASET.$THRESHOLD.$SEED_ID \
                    --device -0
            done
        done
    done
done