from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model, save_model
import os
import argparse

def training(config_path):
    config=read_config(config_path)
    # print(config)
    validation_datasize=config['params']['validation_datasize']
    (X_train, y_train),(X_valid, y_valid),(X_test, y_test) = get_data(validation_datasize)

    #Creating model with Compile
    LOSS_FUNCTION=config['params']['loss_function']
    OPTIMIZER=config['params']['optimizer']
    METRICS=config['params']['metrics']
    NUMB_Classes=config['params']['no_clases']

    model=create_model(LOSS_FUNCTION, OPTIMIZER, METRICS,NUMB_Classes)

    #Training the model
    EPOCHS = config['params']['epochs']
    VALIDATION = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)

    # Here Saving the model
    artifacts_dir=config["artifacts"]["artifacts_dir"]
    model_dir=config["artifacts"]["model_dir"]
    model_name=config["artifacts"]["model_name"]

    #Creating the Model_dir and Artifacts folder
    model_dir_path = os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    save_model(model, model_name, model_dir_path)


if __name__ =='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")
    parsed_args=args.parse_args()
    
    training(config_path=parsed_args.config)