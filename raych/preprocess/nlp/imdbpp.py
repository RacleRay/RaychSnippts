import os
import pandas as pd
import glob
from tqdm import tqdm
from raych.util import logger


def load_data(path, encoding='utf8'):
    """Read set of files from given directory and save returned lines to list.
    """
    file_list = glob.glob(path + '*.txt')
    for file in tqdm(file_list):
        with open(file, 'r', encoding=encoding) as text:
            yield text.read()


def process_imdb(path='aclImdb/', csv_save_path='aclImdb/'):
    # Path to dataset location
    path = path

    # Create a dictionary of paths and lists that store lines (key: value = path: list)
    sets_dict = {'train/pos/': [], 'train/neg/': [],
                'test/pos/': [], 'test/neg/': []}

    # Load the data
    for dataset in tqdm(sets_dict):
        sub_dir = os.path.join(path, dataset)
        sets_dict[dataset] = [text for text in load_data(sub_dir)]

    # Concatenate training and testing examples into one dataset
    dataset = pd.concat([pd.DataFrame({'review': sets_dict['train/pos/'], 'sentiment':1}),
                        pd.DataFrame({'review': sets_dict['test/pos/'], 'sentiment':1}),
                        pd.DataFrame({'review': sets_dict['train/neg/'], 'sentiment':0}),
                        pd.DataFrame({'review': sets_dict['test/neg/'], 'sentiment':0})],
                        axis=0, ignore_index=True)

    dataset.to_csv(csv_save_path, header=False, encoding='utf-8', index=False)

    logger.info(f"[Saved csv file] {csv_save_path}")


if __name__=="__main__":
    pass