import os.path
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUMB_Classes):
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
        tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
        tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
        tf.keras.layers.Dense(NUMB_Classes, activation="softmax", name="outputLayer")
    ]

    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=[METRICS])

    return model_clf  # This will return untrained model

def get_unique_filename(filename):
    unique_filename=time.strftime(f"._%Y%m%d_%H%M%S{filename}")
    return unique_filename

def save_model(model, model_name,model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

def get_unique_plot_filename(filename):
    unique_plot_filename = time.strftime(f"{filename}._%Y%m%d_%H%M%S")
    return unique_plot_filename

def save_plot(filename ,plot_dir,plot):
    unique_plot_filename = get_unique_filename(filename)
    pd.DataFrame(plot).plot(figsize=(10, 7))
    plt.grid(True)
    plt.show()
     # ONLY CREATE IF MODEL_DIR DOES NOT EXISTS
    plotPath = os.path.join(plot_dir, unique_plot_filename)  # model/filename
    plt.savefig(plotPath)
