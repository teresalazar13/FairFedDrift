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
        df = pd.read_csv(f"{path}/list_attr_celeba.csv", delimiter=',', nrows=50000)
        df["Smiling"] = df["Smiling"].replace(-1, 0)
        df["Male"] = df["Male"].replace(-1, 0)
        X = [cv2.imread(f"{path}/img_align_celeba/img_align_celeba/{filename}") for filename in df["image_id"]]
        X = np.array(X)
        y = df["Smiling"]
        s = np.array(df["Male"].tolist())
        index_woman_smiling = df[(df["Male"] == 0) & (df["Smiling"] == 1)].index.tolist()
        index_woman_not_smiling = df[(df["Male"] == 0) & (df["Smiling"] == 0)].index.tolist()
        index_man_smiling = df[(df["Male"] == 1) & (df["Smiling"] == 1)].index.tolist()
        index_man_not_smiling = df[(df["Male"] == 1) & (df["Smiling"] == 0)].index.tolist()

        batched_data = []
        for i in range(self.n_timesteps):
            batched_data_round = []

            for j in range(self.n_clients):
                drift_id = self.drift_ids[i][j]
                if drift_id == 0:
                    iws = random.sample(index_woman_smiling, 50)
                    iwns = random.sample(index_woman_not_smiling, 50)
                    ims = random.sample(index_man_smiling, 50)
                    imns = random.sample(index_man_not_smiling, 10)
                elif drift_id == 1:
                    iws = random.sample(index_woman_smiling, 50)
                    iwns = random.sample(index_woman_not_smiling, 50)
                    ims = random.sample(index_man_smiling, 10)
                    imns = random.sample(index_man_not_smiling, 50)
                else:
                    raise Exception("Drift not supported")
                indexes = np.concatenate([iws, iwns, ims, imns])
                np.random.shuffle(indexes)

                batched_data_round.append([X[indexes], y[indexes], s[indexes], y[indexes]])
                index_woman_smiling = [i for i in index_woman_smiling if i not in iws]
                index_woman_not_smiling = [i for i in index_woman_not_smiling if i not in iwns]
                index_man_smiling = [i for i in index_man_smiling if i not in ims]
                index_man_not_smiling = [i for i in index_man_not_smiling if i not in imns]
            batched_data.append(batched_data_round)

        return batched_data


if __name__ == '__main__':
    FairCeleba()
