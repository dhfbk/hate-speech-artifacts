# Training of baselines (note: step required for data filtering via training dynamics).

# Variables declaration
declare -a PRETRAINED_MODELS=("bert")
declare -a DATASETS=("founta" "gab" "stormfront" "cad")
declare -a SEED_IDS=("1" "2" "3")

# Train baselines using default hyperparameters from previous work
for MODEL in "${PRETRAINED_MODELS[@]}"
do
    for DATASET in "${DATASETS[@]}"
    do
        for SEED_ID in "${SEED_IDS[@]}"
        do
            python machamp/train.py \
                --dataset_config machamp/configs/$DATASET.json \
                --parameters_config machamp/configs/params.$MODEL.$SEED_ID.json \
                --name $MODEL.$DATASET.vanilla.$SEED_ID \
                --device -0
        done
    done
done