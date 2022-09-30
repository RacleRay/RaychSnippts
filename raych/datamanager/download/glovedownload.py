# -*- coding: utf-8 -*-

import os
import urllib.request


class WVDownloader:
    def __init__(self, proxy=None):
        if proxy is not None:
            self.http_proxy = proxy
        else:
            self.http_proxy = "null"

        self.embeding_urls = {
            '42b': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
            '840b': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
            'twitter.27b': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
            '6b': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }

        self.root = ".vector_cache"

    # download module
    def download(self, name="twitter.27b"):
        """Download and unzip an online archive (.zip, .gz, or .tgz).

        Returns:
            dataset_path (str): Path to extracted dataset.
        """
        import zipfile
        import tarfile

        url = self.embeding_urls[name.lower()]

        path = os.path.join(self.root, name)
        filename = os.path.basename(url)
        
        if filename in os.listdir(path):
            return path
        
        zpath = os.path.join(path, filename)
        
        self.download_from_url(url, zpath, self.http_proxy)
        
        ext = os.path.splitext(filename)[-1]
        if ext == '.zip':
            with zipfile.ZipFile(zpath, 'r') as zfile:
                print('extracting')
                zfile.extractall(path)
        elif ext in ['.gz', '.tgz', ".bz2"]:
            with tarfile.open(zpath, 'r:gz') as tar:
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
                    
                
                safe_extract(tar, path=path, members=dirs)

        return path

    @staticmethod
    def download_from_url(url, path, http_proxy):
        if http_proxy != "null":
            proxy = urllib.request.ProxyHandler(
                {'http': http_proxy, 'https': http_proxy})
            # construct a new opener using your proxy settings
            opener = urllib.request.build_opener(proxy)
            # install the openen on the module-level
            urllib.request.install_opener(opener)
            print("proxy in %s" % http_proxy)
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print("!!! BUG", e)
        return path


if __name__ == "__main__":
    dd = WVDownloader()
    path = dd.download("twitter.27b")