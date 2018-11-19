import click
from src.data.dataset import DatasetLoader
from src.evaluation.resnet import build_model
import numpy as np

def evaluate_acc(train_data, test_data, config):
    x_train, y_train = train_data
    x_test, y_test = test_data
    model = build_model(config)
    batch_size  = config['batch_size']
    epochs = config['epochs']
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test ,y_test),
                        shuffle=True)
    return np.max(history.history['val_acc'])

def evaluate_from_data(train_data, test_data, config):
    (x_train, y_train, z_train) = train_data
    (x_test, y_test, z_test) = test_data
    best_y_acc = evaluate_acc((x_train, y_train), (x_test, y_test), config['evaluate_y'])
    best_z_acc = evaluate_acc((x_train, z_train), (x_test, z_test), config['evaluate_z'])
    return best_y_acc, best_z_acc