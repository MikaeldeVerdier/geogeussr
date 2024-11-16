# from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

import configs.training_config as train
import configs.adam_config as adam
from models.losses.root_mean_squared_error import RootMeanSquareError
from dataset_handler import DatasetHandler
from callbacks import ModelCheckpointWithHistory
from countries import *

class Trainer:
    def __init__(self, dataset_path, batch_size, validation_split):
        self.validation_split = validation_split

        self.dataset_handler = DatasetHandler(dataset_path, batch_size)

        # self.log_path = os.path.join(train.SAVE_PATH, "training_log.json")

    """
    def conditional_add_regressor(self, model, encoded_data):
        country_index = encoded_data[0].tolist().index(1)  # np.where(encoded_data[0] == 1)[0]
        if model.specialized_regressors[country_index] is None:
            model.add_regressor(country_index)
    """

    def build_optimizer(self, initial_lr, decay_steps, decay_factor, beta_1, beta_2):
        schedule = ExponentialDecay(initial_lr, decay_steps, decay_factor, staircase=True)
        optimizer = Adam(learning_rate=schedule, beta_1=beta_1, beta_2=beta_2)

        return optimizer

    # def load_metrics(self):
    #     if not os.path.exists(self.log_path):
    #         return {}

    #     with open(self.log_path, "r") as f:
    #         metrics = json.load(f)

    #     return metrics

    # def save_metrics(self, metrics):
    #     with open(self.log_path, "r") as f:
    #         json.dump(metrics, f)

    # def create_csv_logger_callback(self):
    #     csv_logger_callback = CSVLogger(self.log_path, append=True)

    #     return csv_logger_callback

    def prepare_models(self, model):  # Could save this in annotations and then just load it instead of having to do this double...
        optimizer = self.build_optimizer(adam.INITIAL_LEARNING_RATE, adam.DECAY_STEPS, adam.DECAY_FACTOR, adam.BETA_1, adam.BETA_2)
        model.classifier.compile(optimizer=optimizer, loss="categorical_crossentropy")

        for country_name in self.dataset_handler.unique_countries:
            country_index = COUNTRIES.index(country_name)

            if model.specialized_regressors[country_index] is None:
                model.add_regressor(country_index)

            optimizer = self.build_optimizer(adam.INITIAL_LEARNING_RATE, adam.DECAY_STEPS, adam.DECAY_FACTOR, adam.BETA_1, adam.BETA_2)
            model.specialized_regressors[country_index].compile(optimizer=optimizer, loss=[RootMeanSquareError()])

    def create_checkpoint_callback(self, save_freq, name):
        history_filepath = f"{train.SAVE_PATH}/{name}_training_log.json"
        checkpoint_filepath = train.SAVE_PATH + "/{epoch}_" + f"{name}_{train.MODEL_NAME}"  # can't use f-strings because need to format epoch
        model_checkpoint_callback = ModelCheckpointWithHistory(
            history_filepath=history_filepath,
            filepath=checkpoint_filepath,
            verbose=1,
            save_freq=save_freq
        )

        return model_checkpoint_callback

    # def get_start_iteration(self):
        # if not os.path.exists(self.log_path):
        #     return 0

        # start_iteration = len(pd.read_csv(self.log_path))

        # return start_iteration

    def create_generator_split(self, input_shape, preprocess_func, country_name, y_index, split):
        used_batch_size = int(round(self.dataset_handler.batch_size * split))
        if used_batch_size == 0:
            return None

        used_generator = self.dataset_handler.generate_batch(input_shape, preprocess_func, country_name, y_index, used_batch_size)

        return used_generator

    def train(self, model, iteration_amount, save_ratio):  # kinda makes this class redundant when using a generator...
        # metrics = self.load_metrics()
        start_iteration = 0  # self.get_start_iteration()

        # Classifier training
        # checkpoint_callback = self.create_checkpoint_callback(int(iteration_amount * save_ratio), "classifier")

        # train_generator = self.create_generator_split(model.used_input_shape, model.base_process, None, 0, (1 - train.VALIDATION_SPLIT))
        # validation_generator = self.create_generator_split(model.used_input_shape, model.base_process, None, 0, train.VALIDATION_SPLIT)

        # print(f"Training classifier for {iteration_amount} iterations")
        # model.classifier.fit(
        #     train_generator,
        #     epochs=iteration_amount,
        #     callbacks=[checkpoint_callback],
        #     validation_data=validation_generator,
        #     initial_epoch=start_iteration,
        #     steps_per_epoch=1
        # )

        # Regressors training
        for country_name, annotation_count in zip(self.dataset_handler.unique_countries, self.dataset_handler.annotation_counts):
            country_index = COUNTRIES.index(country_name)
            regressor = model.specialized_regressors[country_index]

            checkpoint_callback = self.create_checkpoint_callback(int(iteration_amount * save_ratio), country_name)

            train_generator = self.create_generator_split(model.used_input_shape, model.base_process, country_name, 1, (1 - train.VALIDATION_SPLIT))
            validation_generator = self.create_generator_split(model.used_input_shape, model.base_process, country_name, 1, train.VALIDATION_SPLIT)

            used_iteration_amount = int(iteration_amount * annotation_count / len(self.dataset_handler.annotations))

            print(f"Training regressor ({country_name}) for {used_iteration_amount} iterations")
            regressor.fit(
                train_generator,
                epochs=iteration_amount,
                callbacks=[checkpoint_callback],
                validation_data=validation_generator,
                initial_epoch=used_iteration_amount,
                steps_per_epoch=1
            )
        # metrics |= history.history

        # self.save_metrics(metrics)
