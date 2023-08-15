from typing import Dict

from federated.algorithms.fair_fed_drift.clustering.Clustering import Clustering

"""
    Identity with cluster with the best value of sum of Metrics chosen (M)
"""


class SingleClustering(Clustering):

    def __init__(self):
        name = "single"
        super().__init__(name)
        self.default_weight = 1

    # model_id_amounts -> dict {global_model_id: amount}
    def get_model_weights_for_client(self, results_global_models: Dict, _):
        best_results = []
        best_global_model = None
        model_id_amounts = {}

        for global_model, results in results_global_models.items():
            if sum(results) > sum(best_results):
                best_results = results
                best_global_model = global_model
                model_id_amounts = {global_model.id: 1}

        return best_global_model.model.get_weights(), model_id_amounts
