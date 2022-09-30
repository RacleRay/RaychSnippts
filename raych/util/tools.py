import os
import json
import sys
import gzip
import shutil
import dill
import hashlib
import io
import base58
from typing import Any
from six.moves import urllib
from datetime import datetime
from raych.util import logger
from raych.util.info import show_runtime


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logger.info(" [*] MODEL dir: %s" % args.model_dir)
    logger.info(" [*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def remove_file(path):
    if os.path.exists(path):
        logger.info(" [*] Removed: {}".format(path))
        os.remove(path)


def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    logger.info("[*] {} has backup: {}".format(path, new_path))


def create_dir(dir_path, mode=None):
    """
    Create a directory

    :param dir_path: the directory to create
    :param mode: the permissions to set dir_path to (ie: 0o700)
    """
    if not os.path.exists(dir_path):
        logger.info("[*] Make directories : {}".format(dir_path))
        os.makedirs(dir_path)

    if mode is not None:
        os.chmod(dir_path, mode)


def write_file(file_contents, file_path):
    """
    Write a string or byte string to a file.

    :param file_contents: The string or byte string to write
    :param file_path: The path of the file to write
    """
    file_dir, _ = os.path.split(file_path)
    create_dir(file_dir)

    if isinstance(file_contents, bytes):
        open_kwargs = {}
        file_mode = 'wb'
    else:
        open_kwargs = {'encoding': 'utf-8'}
        file_mode = 'w'

    with open(file_path, file_mode, **open_kwargs) as f:
        f.write(file_contents)


def hash_object(o: Any) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


@show_runtime
def download_and_uncompress(URL, dataset_dir, force=False):
    '''
    Args:
        URL: the download links for data
        dataset_dir: the path to save data
        force: redownload data
    '''
    import zipfile
    import tarfile

    filename = URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    extract_to = os.path.splitext(filepath)[0]

    def download_progress(count, block_size, total_size):
        sys.stdout.write("\r>> Downloading [== %s ==] %.1f%%" % (
            filename, float(count * block_size) / float(total_size) * 100.))
        sys.stdout.flush()

    if not force and os.path.exists(filepath):
        print("file %s already exist" % (filename))
    else:
        filepath, _ = urllib.request.urlretrieve(
            URL, filepath, download_progress)
        print()
        print('Successfully Downloaded', filename)

    if not os.path.exists(extract_to):
        ext = os.path.splitext(filename)[-1]
        if ext == '.zip':
            # with zipfile.ZipFile(filepath) as fd:
            with gzip.open(filepath, 'rb') as f_in, open(extract_to, 'wb') as f_out:
                print('Extracting ', filename)
                shutil.copyfileobj(f_in, f_out)
                print('Successfully extracted')
                print()
            print("Successfully extracted to: ", os.path.join(extract_to, filename))
            return os.path.join(extract_to, filename)

        elif ext in ('.gz', '.tgz', ".bz2"):
            with tarfile.open(filepath, 'r:gz') as tar:
                print('Extracting ', filename)
                dirs = [member for member in tar.getmembers()]
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner) 
                    
                
                safe_extract(tar, path=extract_to, members=dirs)
                print('Successfully extracted')
                print()
            print("Successfully extracted to: ", os.path.join(extract_to, dirs[0].name))
            print("......", os.path.join(extract_to, dirs[0].name))
            return os.path.join(extract_to, dirs[0].name)

    return extract_to


if __name__ == '__main__':
    download_and_uncompress("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                            "/tmp")