import os
import numpy as np

def remove_files(files):
    """
    Remove files from disk

    args: files (str or list) remove all files in 'files'
    """

    if isinstance(files, (list, tuple)):
        for f in files:
            if os.path.isfile(os.path.expanduser(f)):
                os.remove(f)
    elif isinstance(files, str):
        if os.path.isfile(os.path.expanduser(files)):
            os.remove(files)


def create_dir(dirs):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """

    if isinstance(dirs, (list, tuple)):
        for d in dirs:
            if not os.path.exists(os.path.expanduser(d)):
                os.makedirs(d)
    elif isinstance(dirs, str):
        if not os.path.exists(os.path.expanduser(dirs)):
            os.makedirs(dirs)


def setup_logging():
    model_dir = "./saved_model"
    fig_dir = "./images"
    # Create if it does not exist
    create_dir([fig_dir, model_dir])

def accuracy(labels,p_y):
    labels = labels.reshape(-1,)
    p_labels = np.argmax(p_y,axis=1)
    return 100*np.mean(p_labels==labels)