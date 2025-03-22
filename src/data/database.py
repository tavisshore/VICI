from typing import Any, Union

from pathlib import Path
import io
import lmdb
import pickle

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _ascii_encode(data: str) -> bytes:
    return data.encode("ascii")


def _pickle_encode(data: Any, protocol: int) -> bytes:
    return pickle.dumps(data, protocol=protocol)


def _pickle_decode(data: bytes) -> Any:
    return pickle.loads(data)


class Database(object):
    _database = None
    _protocol = None
    _length = None
    _has_fetched = False

    def __init__(
        self,
        path: Union[str, Path],
        readahead: bool = True,
    ):
        """
        Base class for LMDB-backed databases.

        :param path: Path to the database.
        :param readahead: Enables the filesystem readahead mechanism. Useful only if your database fits in RAM.
        """
        if not isinstance(path, str):
            path = str(path)

        self.path = path
        self.readahead = readahead

    @property
    def database(self):
        if self._database is None:
            self._database = lmdb.open(
                path=self.path,
                readonly=True,
                readahead=self.readahead,
                max_spare_txns=256,
                lock=False,
            )
        return self._database

    @database.deleter
    def database(self):
        if self._database is not None:
            self._database.close()
            self._database = None

    @property
    def protocol(self):
        """
        Read the pickle protocol contained in the database.

        :return: The pickle protocol.
        """
        if self._protocol is None:
            self._protocol = self._get(
                key="protocol",
                fencode=_ascii_encode,
                fdecode=_pickle_decode,
            )
        return self._protocol

    @property
    def keys(self):
        """
        Read the keys contained in the database.

        :return: The set of available keys.
        """
        protocol = self.protocol
        keys = self._get(
            key="keys",
            fencode=lambda key: _pickle_encode(key, protocol=protocol),
            fdecode=_pickle_decode,
        )
        return keys

    def __len__(self):
        """
        Returns the number of keys available in the database.

        :return: The number of keys.
        """
        if self._length is None:
            self._length = len(self.keys)
        return self._length

    def __getitem__(self, item):
        """
        Retrieves an item or a list of items from the database.

        :param item: A key or a list of keys.
        :return: A value or a list of values.
        """
        self._has_fetched = True
        if not isinstance(item, list):
            item = self._get(
                key=item,
                fencode=self._fencode,
                fdecode=self._fdecode,
            )
        else:
            item = self._gets(
                keys=item,
                fencodes=self._fencodes,
                fdecodes=self._fdecodes,
            )
        return item

    def _get(self, key, fencode, fdecode):
        """
        Instantiates a transaction and its associated cursor to fetch an item.

        :param key: A key.
        :param fencode:
        :param fdecode:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                key = fencode(key)
                value = cursor.get(key)
                value = fdecode(value)
        self._keep_database()
        return value

    def _gets(self, keys, fencodes, fdecodes):
        """
        Instantiates a transaction and its associated cursor to fetch a list of items.

        :param keys: A list of keys.
        :param fencodes:
        :param fdecodes:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                keys = fencodes(keys)
                _, values = list(zip(*cursor.getmulti(keys)))
                values = fdecodes(values)
        self._keep_database()
        return values

    def _fencode(self, key: Any) -> bytes:
        """
        Converts a key into a byte key.

        :param key: A key.
        :return: A byte key.
        """
        return _pickle_encode(data=key, protocol=self.protocol)

    def _fencodes(self, keys: list[Any]) -> list[bytes]:
        """
        Converts keys into byte keys.

        :param keys: A list of keys.
        :return: A list of byte keys.
        """
        return [self._fencode(key=key) for key in keys]

    def _fdecode(self, value: bytes) -> Any:
        """
        Converts a byte value back into a value.

        :param value: A byte value.
        :return: A value
        """
        return _pickle_decode(data=value)

    def _fdecodes(self, values: list[bytes]) -> list[Any]:
        """
        Converts bytes values back into values.

        :param values: A list of byte values.
        :return: A list of values.
        """
        return [self._fdecode(value=value) for value in values]

    def _keep_database(self):
        """
        Checks if the database must be deleted.

        :return:
        """
        if not self._has_fetched:
            del self.database

    def __iter__(self):
        """
        Provides an iterator over the keys when iterating over the database.

        :return: An iterator on the keys.
        """
        return iter(self.keys)

    def __del__(self):
        """
        Closes the database properly.
        """
        del self.database


class ImageDatabase(Database):
    def _fdecode(self, value: bytes):
        value = io.BytesIO(value)
        image = Image.open(value)
        return image


class LabelDatabase(Database):
    pass


class ArrayDatabase(Database):
    _dtype = None
    _shape = None

    @property
    def dtype(self):
        if self._dtype is None:
            protocol = self.protocol
            self._dtype = self._get(
                key="dtype",
                fencode=lambda key: _pickle_encode(key, protocol=protocol),
                fdecode=_pickle_decode,
            )
        return self._dtype

    @property
    def shape(self):
        if self._shape is None:
            protocol = self.protocol
            self._shape = self._get(
                key="shape",
                fencode=lambda key: _pickle_encode(key, protocol=protocol),
                fdecode=_pickle_decode,
            )
        return self._shape

    def _fdecode(self, value: bytes) -> ndarray:
        return np.frombuffer(value, dtype=self.dtype).reshape(self.shape)

    def _fdecodes(self, values: list[bytes]) -> ndarray:
        shape = (len(values),) + self.shape
        return np.frombuffer(b"".join(values), dtype=self.dtype).reshape(shape)


class TensorDatabase(ArrayDatabase):
    def _fdecode(self, value: bytes) -> Tensor:
        return torch.tensor(super(TensorDatabase, self)._fdecode(value))

    def _fdecodes(self, values: list[bytes]) -> Tensor:
        return torch.tensor(super(TensorDatabase, self)._fdecodes(values))
