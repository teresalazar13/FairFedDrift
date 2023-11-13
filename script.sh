#python3 main.py --fl FedAvg --dataset MNIST-GDrift --varying_disc 0.1
#python3 main.py --fl Oracle --dataset MNIST-GDrift --varying_disc 0.1

python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.1
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.25
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.5
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.75
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 1.0

python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.1 0.1
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.25 0.25
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.5 0.5
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.75 0.75
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 1.0 1.0

python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 0.1
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 0.25
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 0.5
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 0.75
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 1.0

python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 0.1 0.1
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 0.25 0.25
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 0.5 0.5
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 0.75 0.75
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.25 --thresholds 1.0 1.0

python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.1
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.25
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.5
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.75
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 1.0

python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.1 0.1
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.25 0.25
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.5 0.5
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.75 0.75
python3 main.py --fl FairFedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 1.0 1.0
