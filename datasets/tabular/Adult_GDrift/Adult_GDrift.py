from datasets.Feature import Feature
from datasets.tabular.TabularDataset import TabularDataset


class Adult_GDrift(TabularDataset):

    def __init__(self):
        name = "Adult-GDrift"
        input_shape = 14
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
        super().__init__(name, input_shape, sensitive_attribute, target, cat_columns)
