from datasets.Dataset import Dataset
from datasets.Feature import Feature


class Adult(Dataset):

    def __init__(self):
        name = "adult"
        sensitive_attribute = Feature("gender", ["Male"], ["Female"])
        target = Feature("income", ">50K", "<=50K")
        cat_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race",
                       "native-country"]
        super().__init__(name, sensitive_attribute, target, cat_columns)

    def custom_preprocess(self, df):
        return df
