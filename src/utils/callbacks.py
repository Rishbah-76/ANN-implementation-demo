import tensorflow as tf
import numpy as np
import os
import time 

def get_timestamp(tblog_filename):
    timestamp=time.asctime().replace(" ","_").replace(":","_")
    unique_name=f"{tblog_filename}_at_{timestamp}"
    return unique_name


def get_callback(config,X_train):
    logs=config['logs']
    unique_dir_name=get_timestamp("tb_logs")
    tensorboard_root_log=os.path.join(logs["logs_dir"],logs["tensorboard_root_log_dir"],unique_dir_name)
    os.makedirs(tensorboard_root_log, exist_ok=True)

    tensorboard_cb=tf.keras.callbacks.TensorBoard(log_dir=tensorboard_root_log)
    file_writer=tf.summary.create_file_writer(logdir=tensorboard_root_log)
    with file_writer.as_default():
        images=np.reshape(X_train[10:30],(-1,28,28,1))
        tf.summary.image("20 handwritten digit samples", images,max_outputs=25, step=0)

    params=config['params']
    early_stopping=tf.keras.callbacks.EarlyStopping(patience=params['patience'],
                                                    restore_best_weights=params['restore_best_weights'])
    
    artifacts=config['artifacts']
    CKTP_dir=os.path.join(artifacts['artifacts_dir'],artifacts['checkpoints_dir'])
    os.makedirs(CKTP_dir, exist_ok=True)
    CKTP_path=os.path.join(CKTP_dir,"model_ckpt.h5")
    checkpointing_cb=tf.keras.callbacks.ModelCheckpoint(CKTP_path,save_best_only=True)

    return [tensorboard_cb,early_stopping,checkpointing_cb]
                                            