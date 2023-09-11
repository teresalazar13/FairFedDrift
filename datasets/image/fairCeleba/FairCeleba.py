import pandas as pd
import numpy as np
import cv2
import random
from datasets.Dataset import Dataset


class FairCeleba(Dataset):

    def __init__(self):
        name = "fairCeleba"
        is_image = False
        input_shape = (218, 178, 3)
        super().__init__(name, is_image, input_shape)

    def create_batched_data(self, varying_disc):
        path = "./datasets/image/fairCeleba/data"
        df = pd.read_csv(f"{path}/list_attr_celeba.csv", delimiter=',', nrows=20000)
        df = df.sample(frac=1).reset_index(drop=True)
        df["Smiling"] = df["Smiling"].replace(-1, 0)
        df["Male"] = df["Male"].replace(-1, 0)
        df["Black_Hair"] = df["Black_Hair"].replace(-1, 0)
        df["X"] = [cv2.imread(f"{path}/img_align_celeba/img_align_celeba/{filename}") for filename in df["image_id"]]
        df["X"] = [img.astype(np.float32) / 255.0 for img in df["X"]]  # Normalize pixel values to [0, 1]

        df_timestep = np.array_split(df, self.n_timesteps)
        batched_data = []

        for i in range(self.n_timesteps):
            batched_data_round = []
            df_timestep_clients = np.array_split(df_timestep[i], self.n_clients)
            for j in range(self.n_clients):
                df_timestep_client = df_timestep_clients[j]
                drift_id = self.drift_ids[i][j]
                if drift_id == 1:
                    df_timestep_client.loc[
                        (df_timestep_client["Male"] == 1) &
                        (df_timestep_client["Black_Hair"] == 1),
                        "Smiling"
                    ] = 1
                elif drift_id != 0:
                    raise Exception("Drift not supported")
                X = np.array(df_timestep_client["X"].tolist())
                y = df_timestep_client["Smiling"].to_numpy().astype(np.int32)
                s = df_timestep_client["Male"].to_numpy().astype(np.float32)
                batched_data_round.append([X, y, s, y])

            batched_data.append(batched_data_round)

        return batched_data
