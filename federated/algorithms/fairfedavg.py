from tensorflow.keras import backend as K

from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fedavg import get_y, calculate_global_weights
from federated.model import NN_model
from metrics.MetricFactory import get_metrics


class FairFedAvg(Algorithm):

    def __init__(self):
        name = "fair_fedavg"
        super().__init__(name)

    def perform_fl(self, n_timesteps, n_crounds, n_clients, n_features, clients_data, seed, is_image):
        initial_global_model = NN_model(n_features, seed, is_image)
        global_models = [initial_global_model]
        clients_metrics = [get_metrics(is_image) for _ in range(n_clients)]
        client_identities = [[] for _ in range(n_clients)]

        for timestep in range(n_timesteps):
            # STEP 1 - test and determine cluster identities
            cluster_identities, global_models = get_new_cluster_identities(
                clients_data[timestep], global_models, clients_metrics, is_image, n_features, seed, timestep
            )
            print("Cluster identities timestep {} - {}".format(timestep, cluster_identities))
            [client_identities[i].append(id) for i, id in enumerate(cluster_identities)]

            # STEP 2 - Train and average models according to cluster identities
            for cround in range(n_crounds):
                local_weights_lists = [[] for _ in range(len(global_models))]
                client_scaling_factors_lists = [[] for _ in range(len(global_models))]
                for client in range(n_clients):
                    x, y, s = clients_data[timestep][client]
                    cluster_id = cluster_identities[client]
                    global_weights = global_models[cluster_id].get_weights()
                    local_model = NN_model(n_features, seed, is_image)
                    local_model.compile(is_image)
                    local_model.set_weights(global_weights)
                    local_model.learn(x, y)
                    local_weights_lists[cluster_id].append(local_model.get_weights())
                    client_scaling_factors_lists[cluster_id].append(len(x))
                    #K.clear_session()
                    print("Trained model timestep {} cround {} client {} with identity {}".
                          format(timestep, cround, client, cluster_id))

                for i, (weights, scales) in enumerate(zip(local_weights_lists, client_scaling_factors_lists)):
                    if len(weights) > 0:
                        new_global_weights = calculate_global_weights(weights, scales)
                        global_models[i].set_weights(new_global_weights)
                        print("Averaged models on timestep {} cround {} of cluster identity {}".format(timestep, cround, i))
                    else:
                        print("Did not average models on timestep {} cround {} of cluster identity {}".format(timestep, cround,
                                                                                                       i))

        return clients_metrics, client_identities


def get_new_cluster_identities(
        clients_data_timestep, global_models, clients_metrics, is_image, n_features, seed, timestep
):
    cluster_identities = []
    original_size = len(global_models)

    for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
        x, y, s = client_data
        res_models = []
        print("Client {} - checking best model".format(client_id))
        for model_id, global_model in enumerate(global_models[:original_size]):
            pred = global_model.predict(x)
            y_, pred = get_y(y, pred, is_image)
            res_model = []
            print("Client {} - testing on model {}".format(client_id, model_id))
            for client_metric in client_metrics:
                res = client_metric.calculate(y_, pred, s)
                print("{} - {:.2f}".format(client_metric.name, res))
                res_model.append(res)
            res_models.append(res_model)

        cluster_identity = 0
        best_res_sum = 0
        best_res = []
        for j, res in enumerate(res_models):
            if sum(res) >= best_res_sum:
                cluster_identity = j
                best_res_sum = sum(res)
                best_res = res
        pred = global_models[cluster_identity].predict(x)  # save results on best model
        y_, pred = get_y(y, pred, is_image)
        print("Client {} - testing on best model {}".format(client_id, cluster_identity))
        for client_metric in client_metrics:
            res = client_metric.update(y_, pred, s)
            print("{} - {:.2f}".format(client_metric.name, res))

        drift_detected = 0
        for k, res in enumerate(best_res):  # all metrics - accuracy (only balanced accuracy)
            if timestep > 0:
                print("prev", clients_metrics[client_id][k].res[-2], clients_metrics[client_id][k].res[-2] * 0.95, res)
            if timestep > 0 and res < clients_metrics[client_id][k].res[-2] * 0.95:  # DETECT CONCEPT DRIFT  -> prev 0.8 (mnist 0.1) and 0.85 (mnist 0.5)
            #if timestep > 0 and res < 0.85:  # DETECT CONCEPT DRIFT  -> prev 0.8 (mnist 0.1) and 0.85 (mnist 0.5)
                print("Drift detected at client {}".format(client_id))
                global_models.append(NN_model(n_features, seed, is_image))
                cluster_identities.append(len(global_models) - 1)
                drift_detected = 1
                break
        if not drift_detected:
            print("No drift detected at client {}".format(client_id))
            cluster_identities.append(cluster_identity)

    return cluster_identities, global_models
