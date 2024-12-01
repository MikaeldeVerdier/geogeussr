import configs.training_config as train
import configs.full_model_config as model_cfg
from trainer import Trainer
from models.full_model import FullModel

if __name__ == "__main__":
    load = False

    # Create a trainer (always needed)
    trainer = Trainer(train.DATASET_PATH, train.BATCH_SIZE, train.VALIDATION_SPLIT)

    # Create a full model (always needed)
    if not load:
        model = FullModel(model_cfg.IMAGE_SIZE, model_cfg.UNFROZEN_BASE_LAYERS)  # Use a new full model
    else:
        model = FullModel.load_incomplete(train.SAVE_PATH)  # Load a full (incomplete) model

    # Train the full model
    # """
    trainer.train_fullmodel(model, train.AMOUNT_ITERATIONS, train.SAVE_RATIO, load)

    model.save(train.SAVE_PATH)
    # """

    """
    # Train classifier only
    if not load:
        classifier = model.create_classifier()  # Use a new classifier
    else:
        classifier = FullModel.load_submodel(train.SAVE_PATH, "classifier")  # Load a classifier

    trainer.train_classifier(classifier, load, model.used_input_shape, model.base_process, 0, train.AMOUNT_ITERATIONS, train.SAVE_RATIO)
    # model.save(train.SAVE_PATH)
    """

    # Train a regressor only
    """
    country_name = "SWE"
    if not load:
        regressor = model.create_regressor()  # Use a new regressor
    else:
        regressor = FullModel.load_submodel(train.SAVE_PATH, country_name)  # Load a regressor

    trainer.train_regressor(regressor, load, model.used_input_shape, model.base_process, country_name, 0, train.AMOUNT_ITERATIONS, train.SAVE_RATIO)
    # model.save(train.SAVE_PATH)
    """
