# from federated.algorithms.Algorithm import Algorithm, average_weights, get_y
# from federated.algorithms.fair_fed_drift.ClientData import ClientData
# from federated.algorithms.fair_fed_drift.GlobalModels import GlobalModels
# from federated.algorithms.fair_fed_drift.drift_detection.DriftDetectorFactory import get_detector_by_name
# from federated.model import NN_model
# from metrics.MetricFactory import get_metrics, get_metrics_by_names
#
#
# class FairFedDrift(Algorithm):
#     def __init__(self):
#         self.metrics_clustering = None
#         self.drift_detector = None
#         name = "fair_fed_drift"
#         super().__init__(name)
#
#     def set_specs(self, args):
#         self.drift_detector = get_detector_by_name(args.drift_detector)
#         self.drift_detector.set_specs(args)
#         self.metrics_clustering = get_metrics_by_names(args.metrics)
#         metrics_string = "-".join(args.metrics)
#         thresholds_string = "-".join(args.thresholds)
#         super().set_subfolders("{}/drift_detector-{}-{}/metrics-{}".format(
#             self.name, self.drift_detector.name, thresholds_string, metrics_string)
#         )
#
#     def perform_fl(self, seed, clients_data, dataset):
#         global_models = GlobalModels()
#         init_model = get_init_model(dataset, seed)
#         global_model = global_models.create_new_global_model(init_model)
#         for client_id in range(dataset.n_clients):
#             cd = ClientData(clients_data[0][client_id][0], clients_data[0][client_id][1], clients_data[0][client_id][2])
#             global_model.set_client(client_id, cd)
#         clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
#         clients_identities = [[] for _ in range(dataset.n_clients)]
#
#         for timestep in range(dataset.n_timesteps):
#             clients_identities = update_clients_identities(clients_identities, dataset.n_clients, global_models)
#
#             # STEP 1 - Test each client's data on previous clustering identities
#             self.test_models(global_models, clients_data[timestep], clients_metrics, dataset, seed)
#             global_models.reset_clients()
#
#             if timestep != dataset.n_timesteps - 1:
#                 # STEP 2 - Recalculate Global Models (cluster identities) and detect concept drift
#                 global_models = self.update_global_models(clients_data[timestep], global_models, dataset, seed, timestep)
#
#                 # STEP 3 - Merge Global Models
#                 global_models = self.merge_global_models(global_models, dataset, seed)
#
#                 # STEP 4 - Train and average models
#                 global_models = self.train_and_average(clients_data[timestep], global_models, dataset, seed, timestep)
#
#         return clients_metrics, clients_identities
#
#     def train_and_average(self, clients_data_timestep, global_models, dataset, seed, timestep):
#         for cround in range(dataset.n_rounds):
#             local_weights_list = [[] for _ in range(global_models.current_size)]
#             local_scales_list = [[] for _ in range(global_models.current_size)]
#
#             for client_id, client_data in enumerate(clients_data_timestep):
#                 x, y, s, y_original = client_data
#                 local_model, global_model_id = get_model_client(client_id, global_models, dataset, seed)
#                 local_model.learn(x, y)
#                 print("Trained model timestep {} cround {} client {}".format(timestep, cround, client_id))
#                 local_weights_list[global_model_id].append(local_model.get_weights())
#                 local_scales_list[global_model_id].append(len(x))
#
#             for global_model_id, (local_weights, local_scales) in enumerate(zip(local_weights_list, local_scales_list)):
#                 if len(local_weights) > 0:
#                     new_global_weights = average_weights(local_weights, local_scales)
#                     global_models.get_model(global_model_id).model.set_weights(new_global_weights)
#                     print("Averaged models on timestep {} cround {} of cluster {}".format(timestep, cround, global_model_id))
#                 else:
#                     print("Did not average models on timestep {} cround {} of cluster {}".format(timestep, cround, global_model_id))
#
#         return global_models
#
#     def merge_global_models(self, global_models, dataset, seed):
#         size = global_models.current_size
#         if size > 25:
#             raise Exception("Number of global models > 25")
#         all_distances = [[[0 for _ in range(len(self.metrics_clustering))] for _ in range(size)] for _ in range(size)]
#
#         for i in range(len(global_models.models)):
#             for j in range(len(global_models.models)):
#                 id_i = global_models.models[i].id
#                 id_j = global_models.models[j].id
#                 if i != j and len(global_models.models[i].clients.keys()) > 0 and len(global_models.models[j].clients.keys()) > 0:
#                     results_list = []
#                     for client_id in global_models.models[j].clients.keys():
#                         partial_client_data = global_models.models[j].get_partial_client_data(client_id)
#                         results = self.test_client_on_model(
#                             self.metrics_clustering, global_models.models[i].model, partial_client_data,
#                             dataset.is_binary_target
#                         )
#                         results_list.append(results)
#                     worst_results = self.drift_detector.get_worst_results(results_list)
#                     all_distances[id_i][id_j] = worst_results
#
#         distances = [[[0 for _ in range(len(self.metrics_clustering))] for _ in range(size)] for _ in range(size)]
#         for i in range(len(global_models.models)):
#             for j in range(len(global_models.models)):
#                 id_i = global_models.models[i].id
#                 id_j = global_models.models[j].id
#                 results_list = [all_distances[id_i][id_j], all_distances[id_j][id_i]]
#                 worst_results = self.drift_detector.get_worst_results(results_list)
#                 distances[id_i][id_j] = worst_results
#
#         while True:  # While we can still merge global models
#             print_matrix(distances)
#             id_0, id_1 = self.drift_detector.get_next_best_results(distances)
#             if id_0 and id_1:
#                 print("Merged models {} and {}".format(id_0, id_1))
#                 global_models, distances = self.merge_global_models_spec(dataset, seed, global_models, id_0, id_1, distances)
#             else:
#                 return global_models
#
#     def merge_global_models_spec(self, dataset, seed, global_models, id_0, id_1, distances):
#         global_model_0 = global_models.get_model(id_0)
#         global_model_1 = global_models.get_model(id_1)
#         scales = [global_model_0.n_points, global_model_0.n_points]
#         weights = [global_model_0.model.get_weights(), global_model_1.model.get_weights()]
#         new_global_model_weights = average_weights(weights, scales)
#         new_global_model = get_init_model(dataset, seed)
#         new_global_model.set_weights(new_global_model_weights)
#
#         # Create new global Model
#         clients = global_model_0.clients
#         clients.update(global_model_1.clients)
#         new_global_model_created = global_models.create_new_global_model(
#             new_global_model, global_model_0.name, global_model_1.name
#         )
#         for client_id, client_data in clients.items():
#             new_global_model_created.set_client(client_id, client_data)
#
#         # Create new column and row for new model id and update distances
#         new_row = []
#         for i in range(len(distances)):
#             results_list = [distances[id_0][i], distances[id_1][i]]
#             worst_results = self.drift_detector.get_worst_results(results_list)
#             new_row.append(worst_results)
#         for i in range(len(distances)):
#             distances[i].append(new_row[i])
#         new_row.append([0 for _ in range(len(self.metrics_clustering))])
#         distances.append(new_row)
#
#         # Reset Distances of deleted models
#         distances[id_0][id_1] = [0 for _ in range(len(self.metrics_clustering))]
#         distances[id_1][id_0] = [0 for _ in range(len(self.metrics_clustering))]
#         for i in range(len(distances)):
#             distances[id_0][i] = [0 for _ in range(len(self.metrics_clustering))]
#             distances[id_1][i] = [0 for _ in range(len(self.metrics_clustering))]
#             distances[i][id_0] = [0 for _ in range(len(self.metrics_clustering))]
#             distances[i][id_1] = [0 for _ in range(len(self.metrics_clustering))]
#
#         global_models.deleted_merged_model(id_0)
#         global_models.deleted_merged_model(id_1)
#
#         return global_models, distances
#
#
#    def update_global_models(metrics_clustering, drift_detector, clients_data_timestep, global_models, dataset, seed, timestep):
#         global_models_to_create = []
#
#         for client_id, client_data_raw in enumerate(clients_data_timestep):
#             x, y, s, _ = client_data_raw
#             client_data = ClientData(x, y, s)
#
#             # Calculate results on all global models
#             results_global_models = {}
#             for global_model in global_models.models:
#                 results = test_client_on_model(metrics_clustering, global_model.model, client_data, dataset.is_binary_target)
#                 results_global_models[global_model] = results
#
#             # Get Model for client given results_global_models and test
#             model_weights, global_model_id = get_model_weights_for_client(results_global_models)
#             model = get_init_model(dataset, seed)
#             model.set_weights(model_weights)
#             results_model = test_client_on_model(metrics_clustering, model, client_data, dataset.is_binary_target)
#
#             # Detect Drift
#             drift_detected_metrics = drift_detector.drift_detected(results_model, timestep)
#             if sum(drift_detected_metrics) >= 1:
#                 print("Drift detected at client {}".format(client_id))
#                 model = get_init_model(dataset, seed)
#                 global_models_to_create.append([model, client_id, client_data])
#             else:
#                 print("No drift detected at client {}".format(client_id))
#                 global_models.set_client_model(global_model_id, client_id, client_data)
#
#         for model, client_id, client_data in global_models_to_create:
#             new_global_model = global_models.create_new_global_model(model)
#             new_global_model.set_client(client_id, client_data)
#
#         return global_models
#
#     def test_client_on_model(metrics, model, client_data, is_binary_target):
#         results = []
#         pred = model.predict(client_data.x)
#         y_true, y_pred = get_y(client_data.y, pred, is_binary_target)
#         for client_metric in metrics:
#             res = client_metric.calculate(y_true, y_pred, client_data.s)
#             results.append(res)
#
#         return results
#
# def get_init_model(dataset, seed):
#     model = NN_model(dataset, seed)
#     model.compile(dataset)
#
#     return model
#
# def test_models(global_models, clients_data_timestep, clients_metrics, dataset, metrics_clustering, seed):
#     for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
#         x, y, s, _ = client_data
#         model, _ = get_model_client(client_id, global_models, dataset, seed)
#         pred = model.predict(x)
#         y_true, y_pred = get_y(y, pred, dataset.is_binary_target)
#         for client_metric in client_metrics:
#             res = client_metric.update(y_true, y_pred, s)
#             print(res, client_metric.name)
#         for client_metric in metrics_clustering:
#             client_metric.update(y_true, y_pred, s)
#
#
# def update_clients_identities(clients_identities, n_clients, global_models):
#     for client_id in range(n_clients):
#         timestep_client_identities = get_timestep_client_identity(global_models, client_id)
#         clients_identities[client_id].append(timestep_client_identities)
#     print_clients_identities(clients_identities)
#
#     return clients_identities
#
# def get_timestep_client_identity(global_models, client_id):
#     for model in global_models.models:
#         for client in model.clients.keys():
#             if client == client_id:
#                 return model.name
#
#     raise Exception("Model for client {} no found".format(client_id))
#
#
# def print_clients_identities(clients_identities):
#     print("\nClients identities")
#
#     for timestep in range(len(clients_identities[0])):
#         print("\nTimestep ", timestep)
#         identities = {}
#         for client in range(len(clients_identities)):
#             client_identity_timestep = clients_identities[client][timestep]
#             if client_identity_timestep in identities:
#                 identities[client_identity_timestep].append(client)
#             else:
#                 identities[client_identity_timestep] = [client]
#
#         for model_id, clients in identities.items():
#             print("Model id: ", model_id, ":", clients)
#
#
# def print_matrix(matrix):
#     for d in matrix:
#         string = ""
#         for a in d:
#             string += " " + "-".join([str(b) for b in a])
#         print(string)
#
#
# def get_model_weights_for_client(results_global_models):
#     best_results = []
#     best_global_model = None
#     model_id = None
#
#     for global_model, results in results_global_models.items():
#         if sum(results) > sum(best_results) or best_global_model is None:
#             best_results = results
#             best_global_model = global_model
#             model_id = global_model.id
#
#     return best_global_model.model.get_weights(), model_id
#
# def get_model_client(client_id: int, global_models: GlobalModels, dataset, seed):
#     for global_model in global_models.models:
#         for client in global_model.clients:
#             if client == client_id:
#                 weights = global_model.model.get_weights()
#                 model = get_init_model(dataset, seed)
#                 model.set_weights(weights)
#                 return model, global_model.id
#
#     raise Exception("No model for client", client_id)
