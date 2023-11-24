CUDA_VISIBLE_DEVICES=0, python3 main.py --scenario 0 --fl FedAvg --dataset CelebA-GDrift --varying_disc 0.1
CUDA_VISIBLE_DEVICES=0, python3 main.py --scenario 0 --fl Oracle --dataset CelebA-GDrift --varying_disc 0.1
CUDA_VISIBLE_DEVICES=1, python3 main.py --scenario 0 --fl FedAvg --dataset CelebA-GDrift --varying_disc 0.5
CUDA_VISIBLE_DEVICES=1, python3 main.py --scenario 0 --fl Oracle --dataset CelebA-GDrift --varying_disc 0.5

CUDA_VISIBLE_DEVICES=2, python3 main.py --scenario 1 --fl FedAvg --dataset MNIST-GDrift --varying_disc 0.01
CUDA_VISIBLE_DEVICES=2, python3 main.py --scenario 1 --fl Oracle --dataset MNIST-GDrift --varying_disc 0.01
CUDA_VISIBLE_DEVICES=2, python3 main.py --scenario 1 --fl FedAvg --dataset FashionMNIST-GDrift --varying_disc 0.01
CUDA_VISIBLE_DEVICES=2, python3 main.py --scenario 1 --fl Oracle --dataset FashionMNIST-GDrift --varying_disc 0.01

python3 main.py --scenario 1 --fl FedAvg --dataset CIFAR-GDrift --varying_disc 0.01
python3 main.py --scenario 1 --fl Oracle --dataset CIFAR-GDrift --varying_disc 0.01
