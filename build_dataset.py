"""Split the MUMU dataset into train/dev/test sets.

The MuMu dataset comes in the following format:
    AMAZON_ID.jpg
    ...

Original images have various sizes at or below (300, 300).
"""

import argparse
import json
import csv
import random
import os

import urllib.request as req
import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/MUMU',
                    help="Directory with MUMU album images")
parser.add_argument('--output_dir', default='data/300x300_MUMU',
                    help="Where to write the new data")
parser.add_argument('--mumu_metadata', default='data/amazon_metadata_MuMu.json',
                    help="File with MuMu metadata")
parser.add_argument('--data_labels', default='data/MuMu_dataset_multi-label.csv',
                    help="File with MuMu album genres")


def create_dir(dir_name):
    """Create directory with check not to overwrite existing directory"""
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        print("Warning: dir {} already exists".format(dir_name))


def get_image_urls(mumu_metadata):
    """Retrieve all urls for album cover images in the MuMu dataset"""
    with open(mumu_metadata) as f:
        data = json.load(f)
    img_urls = dict()
    for entry in data:
        img_url = entry['imUrl']
        img_id = entry['amazon_id']
        img_urls[img_id] = img_url
    return img_urls


def download_images(img_urls, data_dir):
    """Download album cover images from the MuMu dataset to `data_dir`"""
    create_dir(data_dir)
    for img_id, img_url in tqdm(img_urls.items()):
        img_format = img_url[-4:]
        img_file = os.path.join(data_dir, img_id + img_format)
        req.urlretrieve(img_url, img_file)


def generate_splits(filenames):
    """Generate 80/10/10 train/dev/test splits for data"""
    filenames.sort()
    random.seed(230)
    random.shuffle(filenames)
    split_1 = int(0.8 * len(filenames))
    split_2 = int(0.9 * len(filenames))
    filenames_train = filenames[:split_1]
    filenames_dev = filenames[split_1:split_2]
    filenames_test = filenames[split_2:]
    splits = {'train': filenames_train,
              'dev': filenames_dev,
              'test': filenames_test}
    return splits


def get_album_genres(f_genres):
    """Retrieve genres for each album in the MuMu dataset"""
    album_genres = dict()
    with open(f_genres) as f:
        data = csv.reader(f)
        next(data) # skip header
        for row in data:
            img_id = row[0] # amazon_id
            genres = list(row[5].split(','))
            album_genres[img_id] = genres
    return album_genres


def generate_labels(filenames, album_genres, output_dir, split):
    """Convert genres for each album to binary vector label and save to `output_dir`"""
    # genre_list = list(set([genre for genres in album_genres.values() for genre in genres]))
    genre_list = ['Rock', 'Alternative Rock', 'World Music', 'Dance & Electronic', 'Jazz', 'R&B',
		'Metal', 'Folk', 'Hardcore & Punk', 'Blues', 'Country', 'Latin Music', 'Reggae',
		'Rap & Hip-Hop', 'Oldies', 'Christian', 'Gospel', 'New Age', 'Classical']
    genre_list.sort()
    labels = []
    files = []
    for f in filenames:
        img_id = f.split('/')[-1][:-4]
        genres = album_genres[img_id]
        album_label = [1 if g in genres else 0 for g in genre_list]
        if 1 not in album_label: continue
        labels.append(album_label)
        files.append(f)
    output_file = os.path.join(output_dir, 'y_' + split + '.npy')
    np.save(output_file, np.array(labels))
    return files


def save(filenames, output_dir):
    """Save the images contained in `filenames` to the `output_dir`"""
    for filename in tqdm(filenames):
        image = Image.open(filename)
        image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    # Scrape album artwork if not already present
    if not os.path.isdir(args.data_dir):
        print("Dataset at {} not found. Scraping images now.".format(args.data_dir))
        img_urls = get_image_urls(args.mumu_metadata)
        download_images(img_urls, args.data_dir)

    # Get the filenames in the data directory
    filenames = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.jpg')]

    # Split into 80%/10%/10% train/dev/test
    splits = generate_splits(filenames)

    # Create new output data directory
    create_dir(args.output_dir)

    # Preprocess train, dev and test images
    album_genres = get_album_genres(args.data_labels)
    for split, files in splits.items():
        dir_split = os.path.join(args.output_dir, split)
        create_dir(dir_split)
        output_dir_images = os.path.join(dir_split, 'images')
        create_dir(output_dir_images)
        output_dir_genres = os.path.join(dir_split, 'genres')
        create_dir(output_dir_genres)

        print("Generating {} labels, saving to {}".format(split, output_dir_genres))
        files_present = generate_labels(files, album_genres, output_dir_genres, split)

        print("Processing {} data, saving to {}".format(split, output_dir_images))
        save(files_present, output_dir_images)

    print("Done building dataset")
