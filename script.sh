#python3 main.py --fl FedAvg --dataset MNIST-GDrift --varying_disc 0.5
#python3 main.py --fl Oracle --dataset MNIST-GDrift --varying_disc 0.5

python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.01 --thresholds 0.3
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.01 --thresholds 0.4
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.01 --thresholds 0.5

python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.3
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.4
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.1 --thresholds 0.5

python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.3
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.4
python3 main.py --fl FedDrift --dataset MNIST-GDrift --varying_disc 0.5 --thresholds 0.5
