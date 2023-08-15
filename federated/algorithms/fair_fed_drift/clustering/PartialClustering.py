from typing import Dict

from federated.algorithms.Algorithm import average_weights
from federated.algorithms.fair_fed_drift.clustering.Clustering import Clustering

"""
    Identity with various clusters partially according to sum of Metrics chosen (M)
"""


class PartialClustering(Clustering):

    def __init__(self):
        name = "partial"
        super().__init__(name)
        self.default_weight = 1

    # model_id_amounts -> dict {global_model_id: amount}
    def get_model_weights_for_client(self, results_global_models: Dict, timestep, threshold=0.8):  # TODO
        results_dict = {}
        for global_model, results in results_global_models.items():
            if timestep == 0 or sum(results) >= threshold:
                results_dict[global_model] = sum(results)

        if len(results_dict) == 0:
            print("Len was 0")
            for global_model, results in results_global_models.items():
                results_dict[global_model] = sum(results)

        sum_results = sum(results_dict.values())
        model_id_amounts = {}
        amounts = []
        weights = []
        for model, results in results_dict.items():
            amount = results / sum_results
            model_id_amounts[model.id] = amount
            amounts.append(amount)
            weights.append(model.model.get_weights())

        print(model_id_amounts)

        return average_weights(weights, amounts), model_id_amounts
