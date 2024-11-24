import configs.training_config as train
from trainer import Trainer
from models.full_model import FullModel

if __name__ == "__main__":
    # model = FullModel.load(train.MODEL_PATH)

    load = False

    # Create a trainer (always needed)
    trainer = Trainer(train.DATASET_PATH, train.BATCH_SIZE, train.VALIDATION_SPLIT)

    # Create a full model (always needed)
    if not load:
        model = FullModel()  # Use a new full model
    else:
        model = FullModel.load_self(train.SAVE_PATH, train.MODEL_NAME)  # Load a full model

    # Train the full model
    # """
    trainer.train_fullmodel(model, train.AMOUNT_ITERATIONS, train.SAVE_RATIO, load)

    model.save(train.MODEL_PATH)  # double-saves classifier...
    # """

    """
    # Train classifier only
    if not load:
        classifier = FullModel.create_classifier()  # Use a new classifier
    else:
        classifier = FullModel.load_submodel(train.SAVE_PATH, "classifier", train.MODEL_NAME)  # Load a classifier

    trainer.train_classifier(classifier, load, model.used_input_shape, model.base_process, 0, train.AMOUNT_ITERATIONS, train.SAVE_RATIO)
    """

    # Train a regressor only
    """
    country_name = "SWE"
    if not load:
        regressor = FullModel.create_regressor()  # Use a new regressor
    else:
        regressor = FullModel.load_submodel(train.SAVE_PATH, country_name, train.MODEL_NAME)  # Load a regressor

    trainer.train_regressor(regressor, load, model.used_input_shape, model.base_process, country_name, 0, train.AMOUNT_ITERATIONS, train.SAVE_RATIO)
    """

    # Plot metrics
    """
    from visualizer.metrics_visualizer import MetricsVisualizer

    viz = MetricsVisualizer(train.SAVE_PATH)
    viz.plot_metrics(["classifier", "SWE"], seperate=True)
    """
