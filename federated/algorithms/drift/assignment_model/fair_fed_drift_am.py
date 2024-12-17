import math
import logging
import tensorflow as tf

from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.Identity import Identity
from federated.algorithms.drift.GlobalModels import GlobalModels
from federated.algorithms.drift.assignment_model.autoencoder import Autoencoder
from federated.algorithms.drift.fed_drift import get_init_model, train_and_average, get_clients_data_from_models, \
    test_models, print_clients_identities
from federated.algorithms.drift.assignment_model.assignment_model import AssignmentModel
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged
from metrics.MetricFactory import get_metrics


WORST_CONFIDENCE = 1000


class FairFedDriftAM(Algorithm):

    def __init__(self):
        self.metrics_clustering = [LossPrivileged(), LossUnprivileged()]
        self.thresholds = []
        self.window = None
        name = "FairFedDrift-AM"
        color = "black"
        marker = "o"
        super().__init__(name, color, marker)

    def set_specs(self, args):
        threshold_p = float(args.thresholds[0])
        threshold_up = float(args.thresholds[1])
        self.thresholds = [threshold_p, threshold_up]
        self.window = math.inf
        if args.window:
            self.window = int(args.window)
        super().set_subfolders("{}/window-{}/loss_p-{}/loss_up-{}".format(self.name, self.window, threshold_p, threshold_up))

    def perform_fl(self, seed, clients_data, dataset):
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        global_model = global_models.create_new_global_model(init_model)
        clients_identities = [[Identity(global_model.identity.id, global_model.identity.name)] for _ in range(dataset.n_clients)]
        clients_identities_printing = [[Identity(global_model.identity.id, global_model.identity.name)] for _ in
                                       range(dataset.n_clients)]
        previous_confidence_clients = [[[WORST_CONFIDENCE for _ in range(len(self.metrics_clustering))]] for _ in range(dataset.n_clients)]  # used for drift detection

        # Train with data from first timestep
        #clients_data_models, n_clients_data_models = get_clients_data_from_models(
        #    global_models, clients_identities, clients_data, self.window
        #)
        #global_models = train_and_average(global_models, dataset, seed, 0, clients_data_models)

        # TODO - Train assignment model with data from first timestep
        assignment_model = AssignmentModel(seed)
        assignment_model.compile()
        autoencoder = Autoencoder()
        autoencoder, assignment_model = self.train_assignment_and_autoencoder(autoencoder, assignment_model, clients_data)
        exit()

        for timestep in range(1, dataset.n_timesteps - 1):
            logging.info("Current Global Models")
            for gm in global_models.models:
                logging.info("id: {}, name: {}".format(gm.identity.id, gm.identity.name))

            # STEP 1 - Test each client's data using previous identities
            logging.info("STEP 1 - test (timestep: {})".format(timestep))
            test_models(global_models, clients_data[timestep], clients_metrics, dataset, clients_identities, seed)

            # TODO - STEP 2 - Select most likely model or create new for each client
            #logging.info("STEP 2 - Update (timestep: {})".format(timestep))
            #global_models, clients_identities, previous_loss_clients, clients_new_models = self.update(
            #    clients_data[timestep], global_models, dataset, clients_identities, previous_confidence_clients
            #)

            # STEP 3 - Add models from drifted clients
            #logging.info("STEP 3 - Add models from drifted clients (timestep: {})".format(timestep))
            #for client_id in clients_new_models:
                #new_global_model = global_models.create_new_global_model(get_init_model(dataset, seed))
                #clients_identities[client_id].append(
                    #Identity(new_global_model.identity.id, new_global_model.identity.name))

            # TODO - STEP 4 - Merge Global Models
            #logging.info("STEP 4 - Merge (timestep: {})".format(timestep))
            #global_models, clients_identities = self.merge(clients_data, global_models, dataset, seed, clients_identities)

            # STEP 5 - Train and average models with data from this timestep
            #logging.info("STEP 5 - Train and average (timestep: {})".format(timestep))
            #clients_data_models, n_clients_data_models = get_clients_data_from_models(
            #    global_models, clients_identities, clients_data, self.window
            #)
            #global_models = train_and_average(global_models, dataset, seed, timestep, clients_data_models)

            for client_id in range(dataset.n_clients):
                clients_identities_printing[client_id].append(clients_identities[client_id][-1])

            logging.info("Clients identities (for model) (timestep: {})".format(timestep))
            print_clients_identities(clients_identities)
            # since clients identities change when merging we want to print originals
            logging.info("Clients identities (for printing) (timestep: {})".format(timestep))
            print_clients_identities(clients_identities_printing)

        # test on data from last timestep
        test_models(global_models, clients_data[dataset.n_timesteps - 1], clients_metrics, dataset, clients_identities, seed)

        return clients_metrics, clients_identities_printing

    # TODO
    def train_assignment_and_autoencoder(self, autoencoder, assignment_model, clients_data):
        timestep = 0  # for the first timestep, the concept is 0
        ground_truth = 0  # Initial concept ID
        batch_size = 100
        for client_id in range(len(clients_data[timestep])):
            x, y, _, __ = clients_data[timestep][client_id]
            num_samples = x.shape[0]
            for i in range(0, num_samples, batch_size):
                print(client_id, i, num_samples)
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                # Train autoencoder and generate embeddings
                embeddings = autoencoder.train_on_batch(x_batch, y_batch)
                # Learn concept assignments
                ground_truth = tf.one_hot([0] * batch_size, depth=assignment_model.num_classes)  # One-hot encoded vector for the concept
                assignment_model.learn(embeddings.numpy(), ground_truth)

        return autoencoder, assignment_model

    # TODO
    def update(self,clients_data_timestep, global_models, dataset, clients_identities, previous_confidence_clients):
        #return global_models, clients_identities, previous_loss_clients, clients_new_models
        pass

    # TODO
    def merge(self, clients_data, global_models, dataset, seed, clients_identities):
        #return global_models, clients_identities
        pass
