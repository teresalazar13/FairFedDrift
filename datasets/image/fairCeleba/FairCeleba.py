import pandas as pd
import numpy as np
import cv2
from datasets.Dataset import Dataset


class FairCeleba(Dataset):

    def __init__(self):
        name = "fairCeleba"
        is_image = False
        input_shape = (218, 178, 3)
        super().__init__(name, is_image, input_shape)

    def create_batched_data(self, varying_disc):
        path = "./datasets/image/fairCeleba/data"
        df = pd.read_csv(f"{path}/list_attr_celeba.csv", delimiter=',', nrows=40000)

        df["Smiling"] = df["Smiling"].replace(-1, 0)
        df["Male"] = df["Male"].replace(-1, 0)
        df["No_Beard"] = df["No_Beard"].replace(-1, 0)
        df["X"] = [cv2.imread(f"{path}/img_align_celeba/img_align_celeba/{filename}") for filename in df["image_id"]]
        df["X"] = [img.astype(np.float32) / 255.0 for img in df["X"]]  # Normalize pixel values to [0, 1]

        n_females = int(20000 / (1+varying_disc))
        n_males = int(n_females * varying_disc)
        df_male = df.loc[df['Male'] == 1].copy().sample(n=n_males, random_state=42)
        df_female = df.loc[df['Male'] == 0].copy().sample(n=n_females, random_state=42)

        df_male.reset_index(drop=True, inplace=True)
        df_female.reset_index(drop=True, inplace=True)

        df_timestep_male = np.array_split(df_male, self.n_timesteps)
        df_timestep_female = np.array_split(df_female, self.n_timesteps)
        batched_data = []

        for i in range(self.n_timesteps):
            batched_data_round = []
            df_timestep_clients_male = np.array_split(df_timestep_male[i], self.n_clients)
            df_timestep_clients_female = np.array_split(df_timestep_female[i], self.n_clients)

            for j in range(self.n_clients):
                df_timestep_client_male = df_timestep_clients_male[j]
                df_timestep_client_female = df_timestep_clients_female[j]

                drift_id = self.drift_ids[i][j]
                if drift_id == 1:
                    df_timestep_client_male.loc[
                        (df_timestep_client_male["Male"] == 1) &
                        (df_timestep_client_male["No_Beard"] == 1),
                        "Smiling"
                    ] = 1
                    df_timestep_client_male.loc[
                        (df_timestep_client_male["Male"] == 1) &
                        (df_timestep_client_male["No_Beard"] == 0),
                        "Smiling"
                    ] = 0
                elif drift_id != 0:
                    raise Exception("Drift not supported")
                df_timestep_client = pd.concat([df_timestep_client_male, df_timestep_client_female], axis=0)
                df_timestep_client.sample(frac=1).reset_index(drop=True, inplace=True)  # shuffle
                X = np.array(df_timestep_client["X"].tolist())
                y = df_timestep_client["Smiling"].to_numpy().astype(np.int32)
                s = df_timestep_client["Male"].to_numpy().astype(np.float32)
                batched_data_round.append([X, y, s, y])

            batched_data.append(batched_data_round)

        return batched_data
