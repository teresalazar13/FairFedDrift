#!/bin/bash

#scenarios=(1 2 3)
#thresholds=(0.05 0.1 0.25 0.5 0.75 1.0 1.5)
scenarios=(1)
thresholds=(0.05 0.1)
counter=1

for scenario in "${scenarios[@]}"; do
    screen -dms "$counter"
    screen -x "$counter" CUDA_VISIBLE_DEVICES="$core" python3 main.py --scenario "$scenario" --fl FedAvg --dataset DATASET --varying_disc 0.05
    ((counter++))

    core=$(echo "scale=0; $counter / (50/3)" | bc)
    screen -dms "$counter"
    screen -x "$counter" CUDA_VISIBLE_DEVICES="$core" python3 main.py --scenario "$scenario" --fl Oracle --dataset DATASET --varying_disc 0.05
    ((counter++))

    for threshold in "${thresholds[@]}"; do
        core=$(echo "scale=0; $counter / (50/3)" | bc)

        core=$(echo "scale=0; $counter / (50/3)" | bc)
        screen -dms "$counter"
        screen -x "$counter" CUDA_VISIBLE_DEVICES="$core" python3 main.py --scenario "$scenario" --fl FedDrift --dataset DATASET --thresholds "$threshold" --varying_disc 0.05
        ((counter++))

        core=$(echo "scale=0; $counter / (50/3)" | bc)
        screen -dms "$counter"
        screen -dms "$counter" CUDA_VISIBLE_DEVICES="$core" python3 main.py --scenario "$scenario" --fl FairFedDrift --dataset DATASET --thresholds "$threshold" --varying_disc 0.05
        ((counter++))
    done
done
