import random
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

from datasets.Feature import Feature


class Adult:

    def __init__(self):
        self.name = "adult"
        self.sensitive_attribute = Feature("gender", ["Male"], ["Female"])
        self.target = Feature("income", ">50K", "<=50K")
        self.cat_columns = [
            "workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"
        ]
        self.all_columns = [
            "age", "workclass", "fnlwgt", "education", "educational-num", "marital-status", "occupation",
            "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week", "native-country"
        ]
        self.is_image = False

    def get_folder(self, alg, n_drifts, varying_disc):
        return "./results/{}/n-drifts_{}/disc_{}/{}".format(self.name, n_drifts, varying_disc, alg)

    def get_all_folders(self, n_drifts, varying_disc):
        folder = "./results/{}/n-drifts_{}/disc_{}".format(self.name, n_drifts, varying_disc)
        algs = [x for x in os.listdir(folder) if not x.startswith('.') and "." not in x]
        folders = ["{}/{}".format(folder, x) for x in algs]

        return folder, folders, algs

    def create_batched_data(self, _, n_drifts, varying_disc, n_clients, n_timesteps):
        drift_ids = self.generate_drift_ids(n_clients, n_timesteps, n_drifts)
        batched_data = []
        drift_ids_col = [[] for _ in range(n_clients)]
        df = self.get_dataset()
        dfs_rounds = np.array_split(df, n_timesteps)

        for i in range(n_timesteps):
            batched_data_round = []
            df_round_clients = np.array_split(dfs_rounds[i], n_clients)
            for j in range(n_clients):
                drift_id = drift_ids[i][j]
                df_round_client = df_round_clients[j]
                df_round_client['RAND'] = [random.random() for _ in df_round_client.index]
                if drift_id == 1:
                    df_round_client.loc[
                        (df_round_client["workclass"] == "Private") &
                        (df_round_client["marital-status"] != "Married-civ-spouse") &
                        (df_round_client[self.sensitive_attribute.name] == 1) &
                        (df_round_client[self.target.name] == 0) &
                        (df_round_client["RAND"] > varying_disc),
                        self.target.name
                    ] = 1
                elif drift_id == 2:
                    df_round_client.loc[
                        (df_round_client["marital-status"] == "Married-civ-spouse") &
                        (df_round_client["workclass"] != "Private") &
                        (df_round_client[self.sensitive_attribute.name] == 1) &
                        (df_round_client[self.target.name] == 0) &
                        (df_round_client["RAND"] > varying_disc),
                        self.target.name
                    ] = 1
                df_round_client = df_round_client.drop(columns=['RAND'])
                df_round_client_preprocessed = self.preprocess(df_round_client)
                df_X = df_round_client_preprocessed.copy()
                df_y = df_round_client_preprocessed.pop(self.target.name)
                df_X = df_X.drop(columns=[self.target.name])

                X = df_X.to_numpy().astype(np.float32)
                y = df_y.to_numpy().astype(np.int32)
                s = df_X[self.sensitive_attribute.name].to_numpy().astype(np.float32)
                batched_data_round.append([X, y, s, y])
                drift_ids_col[j].append(drift_ids[i][j])
            batched_data.append(batched_data_round)

        return batched_data, drift_ids_col, len(self.all_columns)

    def generate_drift_ids(self, n_clients, n_rounds, n_drifts):
        drift_ids = [
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
        ]  # start with the same concept (0)

        for i in range(3, n_rounds):
            drift_id_round = []
            for j in range(n_clients):
                if n_drifts > 1 and random.random() > 0.5:  # 50% chance of changing concept
                    choices = list(range(n_drifts))
                    choices.remove(drift_ids[i - 1][j])
                    drift_id = random.choice(choices)
                    print("Drift change at round", i, "client", j)
                else:
                    drift_id = drift_ids[i - 1][j]  # get previous drift id
                drift_id_round.append(drift_id)
            drift_ids.append(drift_id_round)
        print(drift_ids)

        return drift_ids

    def get_dataset(self):
        df = pd.read_csv('./datasets/{}/{}.csv'.format(self.name, self.name))

        s = self.sensitive_attribute
        positive = df[s.name].isin(s.positive)
        negative = df[s.name].isin(s.negative)
        df.loc[positive, s.name] = 1.0
        df.loc[negative, s.name] = 0.0

        positive = df[self.target.name] == self.target.positive
        negative = df[self.target.name] == self.target.negative
        df.loc[positive, self.target.name] = 1.0
        df.loc[negative, self.target.name] = 0.0

        df = df.sample(frac=1).reset_index(drop=True)

        return df

    def preprocess(self, df):
        df[self.cat_columns] = df[self.cat_columns].astype('category')
        df[self.cat_columns] = df[self.cat_columns].apply(lambda x: x.cat.codes)

        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        columns = df.columns
        df = pd.DataFrame(x_scaled)
        df.columns = columns

        return df
