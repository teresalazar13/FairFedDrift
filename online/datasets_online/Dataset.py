from abc import abstractmethod

from sklearn import preprocessing
import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, name, sensitive_attribute, target, cat_columns):
        self.name = name
        self.sensitive_attribute = sensitive_attribute
        self.target = target
        self.cat_columns = cat_columns
        self.is_image = False

    def get_folder(self, alg, n_drifts, varying_disc):
        return "./results_online/{}/{}".format(alg, self.name)

    def create_batched_data(self, _, n_drifts, varying_disc, n_clients, n_rounds):
        # TODO - n_drifts
        X, y, s, n_features = self.preprocess()
        size = len(X) // (n_clients * n_rounds)
        drift_ids_col = [[] for _ in range(n_clients)]
        batched_data = []

        for i in range(n_rounds):
            batched_data_round = []
            for j in range(n_clients):
                X_client = X[i * j + j: i * j + j + size]
                y_client = y[i * j + j: i * j + j + size]
                s_client = s[i * j + j: i * j + j + size]
                batched_data_round.append([X_client, y_client, s_client])
                for _ in range(size):
                    drift_ids_col[j].append(0)
            batched_data.append(batched_data_round)

        return batched_data, drift_ids_col, n_features

    def preprocess(self):
        df = pd.read_csv('./datasets/{}/{}.csv'.format(self.name, self.name)).sample(frac=1)
        df = self.custom_preprocess(df)

        privileged = df[self.sensitive_attribute.name].isin(self.sensitive_attribute.positive)
        unprivileged = df[self.sensitive_attribute.name].isin(self.sensitive_attribute.negative)
        df.loc[privileged, self.sensitive_attribute.name] = 1.0
        df.loc[unprivileged, self.sensitive_attribute.name] = 0.0

        positive = df[self.target.name] == self.target.positive
        negative = df[self.target.name] == self.target.negative
        df.loc[positive, self.target.name] = 1.0
        df.loc[negative, self.target.name] = 0.0

        df[self.cat_columns] = df[self.cat_columns].astype('category')
        df[self.cat_columns] = df[self.cat_columns].apply(lambda x: x.cat.codes)

        # Normalize  -> # TODO - comment this for online
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        columns = df.columns
        df = pd.DataFrame(x_scaled)
        df.columns = columns

        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
        df_X = df.copy()
        df_y = df.pop(self.target.name)
        df_X = df_X.drop(columns=[self.target.name])

        X = df_X.to_numpy().astype(np.float32)
        y = df_y.to_numpy().astype(np.int32)
        s = df_X[self.sensitive_attribute.name].to_numpy().astype(np.float32)
        n_features = len(X[0])

        return X, y, s, n_features

    @abstractmethod
    def custom_preprocess(self, df):
        raise NotImplementedError("Must implement custom_preprocess")
