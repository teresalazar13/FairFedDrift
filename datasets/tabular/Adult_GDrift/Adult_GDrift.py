from datasets.Feature import Feature
from datasets.Dataset import Dataset

import logging
import pandas as pd
import numpy as np
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


class Adult_GDrift(Dataset):

    def __init__(self):
        name = "Adult-GDrift"
        input_shape = (14,)
        sensitive_attribute = Feature("race", ["White"], ["Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
        target = Feature("income", ">50K", "<=50K")
        cat_columns = [
            "workclass", "education", "marital-status", "occupation", "relationship", "gender", "native-country"
        ]
        is_pt = False
        n_classes = 2
        is_image = False
        super().__init__(name, input_shape, is_pt, n_classes, is_image)
        self.sensitive_attribute = sensitive_attribute
        self.target = target
        self.cat_columns = cat_columns


    def create_batched_data(self, varying_disc):
        drift_ids = self.drift_ids
        n_clients = self.n_clients
        n_timesteps = self.n_timesteps
        batched_data = []
        df = self.get_dataset(varying_disc)
        dfs_rounds = np.array_split(df, n_timesteps)

        groups = {
            "Professional": [1/14, 4/14, 10/14, 13/14], # ['Adm-clerical', 'Exec-managerial', 'Prof-specialty', 'Tech-support'],
            "Technical": [2/14, 3/14, 7/14, 5/14, 6/14, 8/14, 9/14, 11/14, 12/14, 14/14], # ['Armed-Forces', 'Craft-repair', 'Machine-op-inspct', 'Farming-fishing', 'Handlers-cleaners', 'Other-service', 'Priv-house-serv', 'Protective-serv', 'Sales', 'Transport-moving'],
        }

        for i in range(n_timesteps):
            batched_data_round = []
            df_round_clients = np.array_split(dfs_rounds[i], n_clients)
            for j in range(n_clients):
                drift_id = drift_ids[i][j]
                df_round_client = df_round_clients[j]

                if drift_id == 1:
                    print("\nDrift 1")
                    print("Ratio : {:.2f}".format(
                        len(df_round_client.loc[(df_round_client[self.sensitive_attribute.name] == 0)
                                                & (df_round_client["gender"] == 1)  # Male
                                                & (df_round_client["occupation"].isin(groups["Professional"]))
                                                & (df_round_client[self.target.name] == 0),
                                                self.target.name]) /
                        len(df_round_client.loc[(df_round_client[self.sensitive_attribute.name] == 0), self.target.name]))
                    )
                    df_round_client.loc[
                        (df_round_client[self.sensitive_attribute.name] == 0)
                        & (df_round_client["gender"] == 1)
                        & (df_round_client["occupation"].isin(groups["Professional"])),
                        self.target.name
                    ] = 1
                elif drift_id == 2:
                    print("\nDrift 2")
                    print("Ratio : {:.2f}".format(
                        len(df_round_client.loc[(df_round_client[self.sensitive_attribute.name] == 0)
                                                & (df_round_client["gender"] == 0)  # Female
                                                & (df_round_client["occupation"].isin(groups["Professional"]))
                                                & (df_round_client[self.target.name] == 0),
                                                self.target.name]) /
                        len(df_round_client.loc[(df_round_client[self.sensitive_attribute.name] == 0), self.target.name]))
                    )
                    df_round_client.loc[
                        (df_round_client[self.sensitive_attribute.name] == 0)
                        & (df_round_client["gender"] == 0)
                        & (df_round_client["occupation"].isin(groups["Professional"])),
                        self.target.name
                    ] = 1
                elif drift_id == 3:
                    print("\nDrift 3")
                    print("Ratio : {:.2f}".format(
                        len(df_round_client.loc[(df_round_client[self.sensitive_attribute.name] == 0)
                                                & (df_round_client["gender"] == 1)
                                                & (df_round_client["occupation"].isin(groups["Technical"]))
                                                & (df_round_client[self.target.name] == 0),
                                                self.target.name]) /
                        len(df_round_client.loc[(df_round_client[self.sensitive_attribute.name] == 0), self.target.name]))
                    )
                    df_round_client.loc[
                        (df_round_client[self.sensitive_attribute.name] == 0)
                        & (df_round_client["gender"] == 1)
                        & (df_round_client["occupation"].isin(groups["Technical"])),
                        self.target.name
                    ] = 1
                elif drift_id == 4:
                    print("\nDrift 4")
                    print("Ratio : {:.2f}".format(
                        len(df_round_client.loc[(df_round_client[self.sensitive_attribute.name] == 0)
                                                & (df_round_client["gender"] == 0)
                                                & (df_round_client["occupation"].isin(groups["Technical"]))
                                                & (df_round_client[self.target.name] == 0),
                                                self.target.name]) /
                        len(df_round_client.loc[(df_round_client[self.sensitive_attribute.name] == 0), self.target.name]))
                    )
                    df_round_client.loc[
                        (df_round_client[self.sensitive_attribute.name] == 0)
                        & (df_round_client["gender"] == 0)
                        & (df_round_client["occupation"].isin(groups["Technical"])),
                        self.target.name
                    ] = 1

                df_X = df_round_client.copy()
                df_y = df_round_client.pop(self.target.name)
                df_X = df_X.drop(columns=[self.target.name])

                X = df_X.to_numpy().astype(np.float32)
                y = df_y.to_numpy().astype(np.int32)
                s = df_X[self.sensitive_attribute.name].to_numpy().astype(np.float32)

                batched_data_round.append([X, y, s, y])
            batched_data.append(batched_data_round)

        return batched_data

    def get_dataset(self, varying_disc):
        df = pd.read_csv('./datasets/tabular/{}/{}.csv'.format(self.name.replace("-", "_"), self.name))
        print(df.head(10)["relationship"])

        s = self.sensitive_attribute
        positive = df[s.name].isin(s.positive)
        negative = df[s.name].isin(s.negative)
        df.loc[positive, s.name] = 1.0
        df.loc[negative, s.name] = 0.0

        positive = df[self.target.name] == self.target.positive
        negative = df[self.target.name] == self.target.negative
        df.loc[positive, self.target.name] = 1.0
        df.loc[negative, self.target.name] = 0.0

        df[self.cat_columns] = df[self.cat_columns].astype('category')
        for col in self.cat_columns:
            print("Category mapping for column:", col)
            print(dict(enumerate(df[col].cat.categories)))
        df[self.cat_columns] = df[self.cat_columns].apply(lambda x: x.cat.codes)

        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        columns = df.columns
        df = pd.DataFrame(x_scaled)
        df.columns = columns

        size_priv = len(
            df.loc[
                (df[self.sensitive_attribute.name] == 1),
                self.target.name
            ]
        )
        size_unpriv = len(
            df.loc[
                (df[self.sensitive_attribute.name] == 0),
                self.target.name
            ]
        )
        logging.info("Size priv {}, size unpriv {}, div {}".format(size_priv, size_unpriv, size_unpriv/size_priv))

        if varying_disc != 0.0:
            if size_unpriv < size_priv * varying_disc:
                n = int(size_priv * varying_disc) - size_unpriv
                logging.info("Adding Unprivileged Instances: {}".format(n))
                sampling_strategy = {0: size_unpriv + n, 1: size_priv}
                df = self.oversample(df, sampling_strategy)
            else:
                n = int(size_unpriv / varying_disc) - size_priv
                logging.info("Adding Privileged Instances: {}".format(n))
                sampling_strategy = {0: size_unpriv, 1: size_priv + n}
                df = self.oversample(df, sampling_strategy)

        new_size_priv = len(
            df.loc[
                (df[self.sensitive_attribute.name] == 1),
                self.target.name
            ]
        )
        new_size_unpriv = len(
            df.loc[
                (df[self.sensitive_attribute.name] == 0),
                self.target.name
            ]
        )
        logging.info("New size priv {}, new size unpriv {}, div {}".format(new_size_priv, new_size_unpriv, new_size_unpriv/new_size_priv))

        df = df.sample(frac=1).reset_index(drop=True)

        return df


    def oversample(self, df, sampling_strategy):
        X = df.drop(self.sensitive_attribute.name, axis=1)
        y = df[self.sensitive_attribute.name]
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X_resampled[self.sensitive_attribute.name] = y_resampled

        return X_resampled
