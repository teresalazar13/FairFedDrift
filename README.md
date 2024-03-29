## FairFedDrift

## General Structure

The main code provided in `federated` and `metrics`.
The datasets used in this study are provided in `datasets`.
Results generated by the code are saved in the results folder.

## Dependencies

You'll need a working Python environment to run the code.
The required dependencies are specified in the file `requirements.txt`.

You can install all required dependencies by running:

    pip install -r requirements.txt

## Reproducing the results

To build and test the software and produce all results run this in the top level of the repository:

    python3 main.py --scenario SCENARIO --fl ALGORITHM --dataset DATASET --varying_disc ALPHA [--thresholds THRESHOLDS] [--window WINDOW]

## Reproducing the Results

To build and test the software and produce all results, run the following command in the top level of the repository:

    python3 read_all.py --scenarios SCENARIOS --dataset DATASET --varying_disc ALPHA [--window WINDOW]

## License

All source code is made available under a Creative Commons License license. (https://creativecommons.org/licenses/by/4.0/)
