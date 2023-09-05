import random
import pandas as pd
import numpy as np
from sklearn import preprocessing

from datasets.Dataset import Dataset


class TabularDataset(Dataset):

    def __init__(self, name, input_shape, sensitive_attribute, target, cat_columns):
        is_image = False
        super().__init__(name, is_image, input_shape)
        self.sensitive_attribute = sensitive_attribute
        self.target = target
        self.cat_columns = cat_columns

    def create_batched_data(self, varying_disc):
        drift_ids = self.drift_ids
        n_clients = self.n_clients
        n_timesteps = self.n_timesteps
        batched_data = []
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
                        (df_round_client[self.sensitive_attribute.name] == 0) &
                        (df_round_client["hours-per-week"] <= 40),
                        self.target.name
                    ] = 0
                """
                if drift_id == 1:
                    df_round_client.loc[
                        (df_round_client[self.sensitive_attribute.name] == 1) &
                        (df_round_client[self.target.name] == 1) &
                        (df_round_client["RAND"] > varying_disc),
                        self.target.name
                    ] = 0
                    df_round_client.loc[
                        (df_round_client[self.sensitive_attribute.name] == 0) &
                        (df_round_client[self.target.name] == 0) &
                        (df_round_client["RAND"] > varying_disc),
                        self.target.name
                    ] = 1
                elif drift_id == 2:
                    df_round_client.loc[
                        (df_round_client[self.sensitive_attribute.name] == 1) &
                        (df_round_client[self.target.name] == 0) &
                        (df_round_client["RAND"] > varying_disc),
                        self.target.name
                    ] = 1
                    df_round_client.loc[
                        (df_round_client[self.sensitive_attribute.name] == 0) &
                        (df_round_client[self.target.name] == 1) &
                        (df_round_client["RAND"] > varying_disc),
                        self.target.name
                    ] = 0"""
                df_round_client = df_round_client.drop(columns=['RAND'])
                df_round_client_preprocessed = self.preprocess(df_round_client)
                df_X = df_round_client_preprocessed.copy()
                df_y = df_round_client_preprocessed.pop(self.target.name)
                df_X = df_X.drop(columns=[self.target.name])

                X = df_X.to_numpy().astype(np.float32)
                y = df_y.to_numpy().astype(np.int32)
                s = df_X[self.sensitive_attribute.name].to_numpy().astype(np.float32)

                batched_data_round.append([X, y, s, y])
            batched_data.append(batched_data_round)

        return batched_data

    def get_dataset(self):
        df = pd.read_csv('./datasets/tabular/{}/{}.csv'.format(self.name, self.name))
        df = pd.concat([df, df, df, df, df], ignore_index=True)  # TODO

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
