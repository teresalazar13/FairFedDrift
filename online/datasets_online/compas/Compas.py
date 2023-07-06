from datasets.Dataset import Dataset
from datasets.Feature import Feature


class Compas(Dataset):

    def __init__(self):
        name = "compas"
        sensitive_attribute = Feature("race", ["Caucasian"], ["African-American", "Hispanic", "Other", "Asian", "Native American"])
        target = Feature("two_year_recid", 0, 1)
        cat_columns = ["sex", "c_charge_degree", "c_charge_desc"]
        super().__init__(name, sensitive_attribute, target, cat_columns)

    def custom_preprocess(self, df):
        return df.drop(["sex-race", "age_cat", "score_text"], axis=1)
