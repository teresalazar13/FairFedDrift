python3 main.py --fl FedDrift --dataset CIFAR-GDrift --varying_disc 0.25 --thresholds 0.5  # arya
python3 main.py --fl FairFedDrift --dataset CIFAR-GDrift --varying_disc 0.75 --thresholds 1.0 1.0 # cvd-4
python3 main.py --fl FedDrift --dataset CIFAR-GDrift --varying_disc 0.75 --thresholds 1.0 # cvd-2

python3 main.py --fl FedDrift --dataset CIFAR-GDrift --varying_disc 0.1 --thresholds 0.1 # crai

