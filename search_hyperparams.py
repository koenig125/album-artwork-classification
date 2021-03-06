"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

from model.utils import Params

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/300x300_MUMU',
                    help="Directory containing the dataset")
parser.add_argument('--param', default='learning_rate',
                    help="Type of hyperparameter to search over")


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir}".format(python=PYTHON,
            model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch over one parameter
    if args.param == 'channel':
        channels = [16, 32, 64]
        for channel in channels:
            params.num_channels = channel
            job_name = "channel_{}".format(channel)
            launch_training_job(args.parent_dir, args.data_dir, job_name, params)
    elif args.param == 'learning_rate':
        learning_rates = [1e-5, 1e-4, 1e-3]
        for learning_rate in learning_rates:
            params.learning_rate = learning_rate
            job_name = "learning_rate_{}".format(learning_rate)
            launch_training_job(args.parent_dir, args.data_dir, job_name, params)
    elif args.param == 'regularization_rate':
        regularization_rates = [.0001, .001, .01, .1]
        for regularization_rate in regularization_rates:
            params.regularization_rate = regularization_rate
            job_name = "regularization_rate_{}".format(regularization_rate)
            launch_training_job(args.parent_dir, args.data_dir, job_name, params)
    elif args.param == 'dropout_rate':
        dropout_rates = [.1, .2, .5]
        for dropout_rate in dropout_rates:
            params.dropout_rate = dropout_rate
            job_name = "dropout_rate_{}".format(dropout_rate)
            launch_training_job(args.parent_dir, args.data_dir, job_name, params)
