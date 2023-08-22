from datasets.Feature import Feature
from datasets.tabular.TabularDataset import TabularDataset


class Adult(TabularDataset):

    def __init__(self):
        name = "adult"
        n_features = 14
        sensitive_attribute = Feature("race", ["White"], ["Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
        # self.sensitive_attribute = Feature("gender", ["Male"], ["Female"])
        target = Feature("income", ">50K", "<=50K")
        cat_columns = [
            "workclass", "education", "marital-status", "occupation", "relationship", "gender", "native-country"
        ]
        """
        cat_columns = [
            "workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"
        ]"""
        super().__init__(name, n_features, sensitive_attribute, target, cat_columns)
