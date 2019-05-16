"""Train the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.training import train_and_evaluate
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/300x300_MUMU',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproducible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment (comment out if developing model)
    # model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    # overwritting = model_dir_has_best_weights and args.restore_from is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    train_images_dir = os.path.join(data_dir, "train/images")
    dev_images_dir = os.path.join(data_dir, "dev/images")
    train_genres_file = os.path.join(data_dir, "train/genres/y_train.npy")
    dev_genres_file = os.path.join(data_dir, "dev/genres/y_dev.npy")

    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_images_dir, f) for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
    eval_filenames = [os.path.join(dev_images_dir, f) for f in os.listdir(dev_images_dir) if f.endswith('.jpg')]

    # Labels will be binary vector representing all genres
    train_labels = np.load(train_genres_file)
    eval_labels = np.load(dev_genres_file)

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_filenames, train_labels, params)
    eval_inputs = input_fn(False, eval_filenames, eval_labels, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
