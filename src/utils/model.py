import tensorflow as tf

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICSnumber_classes):
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(number_classes, activation="softmax", name="outputLayer")
          ]
    
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()


    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=[METRICS])
        
    return model_clf  #This wwill return untrained model