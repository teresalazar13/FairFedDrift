import os


def check(file_path, file_path_2):
    check = True
    if not os.path.exists(file_path):
        if os.path.exists(file_path_2):
            with open(file_path_2, 'r') as f:
                lines = f.readlines()
                if lines and lines[-1].strip() == "INFO:root:Number of global models > 30":
                    check = True
                else:
                    check = False
        else:
            check = False

    if check:
        #print("OK", file_path)
        pass
    else:
        print("\nNOOOOOOOT OK", file_path)


# List of t values
s_values = [1, 2, 3, 4, 5]
disc_values = [0.05, 0.1]
t_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# Base path
base_path_FedAvg = "results/scenario-{s}/Adult-GDrift/disc_{disc}/FedAvg/client_1/results.csv"
base_path_FedAvg_2 = "results/scenario-{s}/Adult-GDrift/disc_{disc}/FedAvg/output.txt"
base_path_Oracle = "results/scenario-{s}/Adult-GDrift/disc_{disc}/Oracle/client_1/results.csv"
base_path_Oracle_2 = "results/scenario-{s}/Adult-GDrift/disc_{disc}/Oracle/output.txt"
base_path_FedDrift = "results/scenario-{s}/Adult-GDrift/disc_{disc}/FedDrift/window-inf/loss-{t}/client_1/results.csv"
base_path_FedDrift_2 = "results/scenario-{s}/Adult-GDrift/disc_{disc}/FedDrift/window-inf/loss-{t}/output.txt"
base_path_FairFedDrift = "results/scenario-{s}/Adult-GDrift/disc_{disc}/FairFedDrift/window-inf/loss_p-{t}/loss_up-{t}/client_1/results.csv"
base_path_FairFedDrift_2 = "results/scenario-{s}/Adult-GDrift/disc_{disc}/FairFedDrift/window-inf/loss_p-{t}/loss_up-{t}/output.txt"

for s in s_values:
    for disc in disc_values:
        check(base_path_FedAvg.format(s=s, disc=disc), base_path_FedAvg_2.format(s=s, disc=disc))
        check(base_path_Oracle.format(s=s, disc=disc), base_path_Oracle_2.format(s=s, disc=disc))
        for t in t_values:
            check(base_path_FedDrift.format(s=s, disc=disc, t=t), base_path_FedDrift_2.format(s=s, disc=disc, t=t))
            check(base_path_FairFedDrift.format(s=s, disc=disc, t=t), base_path_FairFedDrift_2.format(s=s, disc=disc, t=t))


