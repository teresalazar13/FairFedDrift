#!/bin/bash

DATASET="$1"
ALPHA="$2"
CSV="$3"

# Read the CSV file line by line
while IFS=, read -r screen_command; do
    # Extract screen name and command from each line
    screen_name=$(echo "$screen_command" | cut -d',' -f1)
    command_to_run=$(echo "$screen_command" | cut -d',' -f2-)

    # Replace DATASET and ALPHA in the command
    command_to_run="${command_to_run//DATASET/$DATASET}"
    command_to_run="${command_to_run//ALPHA/$ALPHA}"

    # Create a screen session with the screen name and run the command
    screen -dmS "$screen_name" bash -c "$command_to_run"

done < CSV