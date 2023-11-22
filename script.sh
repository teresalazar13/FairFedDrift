python3 main.py --scenario 3 --fl FedAvg --dataset FashionMNIST-GDrift --varying_disc 0.5
python3 main.py --scenario 3 --fl Oracle --dataset FashionMNIST-GDrift --varying_disc 0.5
python3 main.py --scenario 3 --fl FedDrift --dataset FashionMNIST-GDrift --thresholds 0.1 --varying_disc 0.5
python3 main.py --scenario 3 --fl FedDrift --dataset FashionMNIST-GDrift --thresholds 0.25 --varying_disc 0.5
python3 main.py --scenario 3 --fl FedDrift --dataset FashionMNIST-GDrift --thresholds 0.5 --varying_disc 0.5
python3 main.py --scenario 3 --fl FedDrift --dataset FashionMNIST-GDrift --thresholds 0.75 --varying_disc 0.5
python3 main.py --scenario 3 --fl FedDrift --dataset FashionMNIST-GDrift --thresholds 1.0 --varying_disc 0.5
python3 main.py --scenario 3 --fl FairFedDrift --dataset FashionMNIST-GDrift --thresholds 0.1 0.1 --varying_disc 0.5
python3 main.py --scenario 3 --fl FairFedDrift --dataset FashionMNIST-GDrift --thresholds 0.25 0.25 --varying_disc 0.5
python3 main.py --scenario 3 --fl FairFedDrift --dataset FashionMNIST-GDrift --thresholds 0.5 0.5 --varying_disc 0.5
python3 main.py --scenario 3 --fl FairFedDrift --dataset FashionMNIST-GDrift --thresholds 0.75 0.75 --varying_disc 0.5
python3 main.py --scenario 3 --fl FairFedDrift --dataset FashionMNIST-GDrift --thresholds 1.0 1.0 --varying_disc 0.5
