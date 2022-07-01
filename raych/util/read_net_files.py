import csv
import urllib.request
import codecs
from PIL import Image
import requests
from io import BytesIO


def read_csv(csv_url):
    urlstream = urllib.request.urlopen(csv_url)
    csvfile = csv.reader(codecs.iterdecode(urlstream, 'utf-8'))
    # next(csvfile) # Skip header row
    return csvfile  # iterable


def read_txt(txt_url):
    with urllib.request.urlopen(txt_url) as urlstream:
        for line in codecs.iterdecode(urlstream, 'utf-8'):
            yield line


def read_image(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    return img