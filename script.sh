#python3 main.py --fl Oracle --dataset CIFAR-GDrift --varying_disc 0.5  # local
#python3 main.py --fl FedDrift --dataset CIFAR-GDrift --varying_disc 0.5 --thresholds 1.0  # cvd
#python3 main.py --fl FairFedDrift --dataset CIFAR-GDrift --varying_disc 0.5 --thresholds 1.0 1.0  # arya

python3 main.py --fl FedDrift --dataset CIFAR-GDrift --varying_disc 0.25 --thresholds 0.5  # local 2
python3 main.py --fl FedDrift --dataset CIFAR-GDrift --varying_disc 0.25 --thresholds 1.0 1.0  # arya 2
python3 main.py --fl FedAvg --dataset CIFAR-GDrift --varying_disc 0.75  # cvd 2

python3 main.py --fl Oracle --dataset CIFAR-GDrift --varying_disc 0.75  # arya 3
