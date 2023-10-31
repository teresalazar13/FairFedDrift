from federated.algorithms.Algorithm import Algorithm, average_weights, get_y
from federated.model import NN_model
from metrics.MetricFactory import get_metrics


# Has access to client identities and clusters models based on that
class Oracle(Algorithm):

    def __init__(self):
        name = "oracle"
        super().__init__(name)

    def perform_fl(self, seed, clients_data, dataset):
        global_models = []
        for i in range(dataset.n_drifts):
            global_models.append(NN_model(dataset, seed))
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        clients_identities = [[] for _ in range(dataset.n_clients)]

        for timestep in range(dataset.n_timesteps):
            for client_id in range(dataset.n_clients):  # Update client identities
                clients_identities[client_id].append([dataset.drift_ids[timestep][client_id], 1])

            # STEP 1 - Test
            test_models(global_models, clients_data[timestep], clients_metrics, dataset, timestep)

            # STEP 2 - Train and average models
            for cround in range(dataset.n_rounds):
                local_weights_list = [[] for _ in range(len(global_models))]
                local_scales_list = [[] for _ in range(len(global_models))]
                for client in range(dataset.n_clients):
                    x, y, s, _ = clients_data[timestep][client]
                    id = dataset.drift_ids[timestep][client]
                    global_weights = global_models[id].get_weights()
                    local_model = NN_model(dataset, seed)
                    local_model.compile(dataset)
                    local_model.set_weights(global_weights)
                    local_model.learn(x, y)
                    local_weights_list[id].append(local_model.get_weights())
                    local_scales_list[id].append(len(x))
                    #K.clear_session()
                    print("Trained model timestep {} cround {} client {}".format(timestep, cround, client))

                for i in range(len(global_models)):
                    if len(local_weights_list[i]) > 0:
                        new_global_weights = average_weights(local_weights_list[i], local_scales_list[i])
                        global_models[i].set_weights(new_global_weights)
                        print("Averaged models on timestep {} cround {} of cluster {}".format(timestep, cround, i))
                    else:
                        print("Did not average models on timestep {} cround {} of cluster {}".format(timestep, cround, i))

        client_identities = [[] for _ in range(dataset.n_clients)]
        for i in range(dataset.n_clients):
            for _ in range(dataset.n_timesteps):
                client_identities[i].append(0)

        return clients_metrics, client_identities

def test_models(global_models, clients_data_timestep, clients_metrics, dataset, timestep):
    for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
        x, y_true_raw, s, _ = client_data
        id = dataset.drift_ids[timestep][client_id]
        model = global_models[id]
        y_pred_raw = model.predict(x)
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, dataset.is_binary_target)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            print(res, client_metric.name)
