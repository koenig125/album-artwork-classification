"""Evaluate the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/300x300_MUMU',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_images_dir = os.path.join(data_dir, "test/images")
    test_genres_file = os.path.join(data_dir, "test/genres/y_test.npy")

    # Get the filenames from the test set
    test_filenames = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.endswith('.jpg')]

    # Labels will be binary vector representing all genres
    test_labels = np.load(test_genres_file)

    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    # Evaluate the model
    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
