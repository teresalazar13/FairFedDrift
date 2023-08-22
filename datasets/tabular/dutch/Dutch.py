from datasets.Feature import Feature
from datasets.tabular.TabularDataset import TabularDataset


class Dutch(TabularDataset):

    def __init__(self):
        name = "dutch"
        n_features = 11
        sensitive_attribute = Feature("sex", ["male"], ["female"])
        target = Feature("occupation", 1, 0)
        cat_columns = []
        super().__init__(name, n_features, sensitive_attribute, target, cat_columns)
