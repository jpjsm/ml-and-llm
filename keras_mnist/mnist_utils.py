import os
import functools
import operator
import gzip
import struct
import array
import tempfile
import urllib.request
from urllib.parse import urljoin
from pathlib import Path
import numpy


__version__ = '0.2.2'


# `datasets_url` and `temporary_dir` can be set by the user using:
# >>> mnist.datasets_url = 'http://my.mnist.url'
# >>> mnist.temporary_dir = lambda: '/tmp/mnist'
datasets_url = 'https://www.kaggle.com/datasets/hojjatk/mnist-dataset' #'https://kdd.ics.uci.edu/databases/mnist/' # 'http://yann.lecun.com/exdb/mnist/'
temporary_dir = tempfile.gettempdir


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


def download_file(fname, target_dir=None, force=False):
    """Download fname from the datasets_url, and save it to target_dir,
    unless the file already exists, and force is False.

    Parameters
    ----------
    fname : str
        Name of the file to download

    target_dir : str
        Directory where to store the file

    force : bool
        Force downloading the file, if it already exists

    Returns
    -------
    fname : str
        Full path of the downloaded file
    """
    target_dir = target_dir or temporary_dir()
    target_fname = str(Path(target_dir).joinpath(fname).absolute())
    print(f"{target_fname} exists: {Path(target_fname).exists()}")
    if force or not os.path.isfile(target_fname):
        assert not Path(target_fname).exists()
        url = urljoin(datasets_url, fname)

        try:
          # Read the file inside the .gz archive located at url
          with urllib.request.urlopen(url) as response:
             with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()

          # write to file in binary mode 'wb'
          with open(target_fname, 'wb') as f:
             f.write(file_content)
        except Exception as e:
            print(f"Exception msg: {e},\n\t{url=},\n\t{target_dir=},\n\t{target_fname=},")
            raise

    return target_fname


def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.

    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse

    endian : str
        Byte order of the IDX file. See [1] for available options

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file

    1. https://docs.python.org/3/library/struct.html
        #byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, '
                             'file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, '
                             'file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type '
                             '0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items,
                                                          len(data)))

    return numpy.array(data).reshape(dimension_sizes)


def download_and_parse_mnist_file(fname, target_dir=None, force=False):
    """Download the IDX file named fname from the URL specified in dataset_url
    and return it as a numpy array.

    Parameters
    ----------
    fname : str
        File name to download and parse

    target_dir : str
        Directory where to store the file

    force : bool
        Force downloading the file, if it already exists

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    """
    fname = download_file(fname, target_dir=target_dir, force=force)
    print(f"[download_and_parse_mnist_file] {fname=}, extension: {os.path.splitext(fname)[1]}")
    fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
    with fopen(fname, 'rb') as fd:
        return parse_idx(fd)


def train_images():
    """Return train images from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.

    Returns
    -------
    train_images : numpy.ndarray
        Numpy array with the images in the train MNIST database. The first
        dimension indexes each sample, while the other two index rows and
        columns of the image
    """
    return download_and_parse_mnist_file('train-images-idx3-ubyte.gz')


def test_images():
    """Return test images from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.

    Returns
    -------
    test_images : numpy.ndarray
        Numpy array with the images in the train MNIST database. The first
        dimension indexes each sample, while the other two index rows and
        columns of the image
    """
    return download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')


def train_labels():
    """Return train labels from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.

    Returns
    -------
    train_labels : numpy.ndarray
        Numpy array with the labels 0 to 9 in the train MNIST database.
    """
    return download_and_parse_mnist_file('train-labels-idx1-ubyte.gz')


def test_labels():
    """Return test labels from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.

    Returns
    -------
    test_labels : numpy.ndarray
        Numpy array with the labels 0 to 9 in the train MNIST database.
    """
    return download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')