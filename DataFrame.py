#  Copyright (c) 2021 Jäger. All rights reserved.

from typing import List
from numpy import str, random, array, arange, genfromtxt, ndarray, asarray, empty


class DataFrame:
    col_names: ndarray
    data: ndarray

    def __init__(self, data: ndarray = None, col_names: ndarray = None):
        self.col_names: ndarray = empty(0)
        self.data: ndarray = empty((0, 0))
        if data is not None:
            self.data = data
            if col_names is not None:
                if len(col_names) != len(self.data):
                    raise Exception('Columns names length mismatch')
                else:
                    self.col_names = asarray(col_names)
                    pass
                pass
            pass
        pass

    def get_col(self, col: int) -> ndarray:
        """
        Use index to get a specific column.

        Parameters
        ----------
        col: int
            Position index of the column.

        Returns
        -------
        ndarray
            The column at this position.
        """
        return self.data[:, col]

    def get_row(self, row: int) -> ndarray:
        """
        Use index to get a specific row.

        Parameters
        ----------
        row: int
            Position index of the row.

        Returns
        -------
        ndarray
            The row at this position.
        """
        return self.data[row]

    def get_cols(self, cols: List[int]) -> ndarray:
        """
        Use a list of indices to get a new DataFrame consists of specific columns.
        Parameters
        ----------
        cols: List[int]
            Position indices of the columns.

        Returns
        -------
        ndarray
            A DataFrame with these columns.
        """
        return self.data[:, cols]

    def get_rows(self, rows: List[int]) -> ndarray:
        """
        Use a list of indices to get a new DataFrame consists of specific rows.

        Parameters
        ----------
        rows: List[int]
            Position indices of the rows.

        Returns
        -------
        ndarray
            A DataFrame with these rows.
        """
        return self.data[rows]

    def get_sub_df(self, rows: List[int], cols: List[int]) -> ndarray:
        """
        Use two lists of indices to get a new DataFrame consists of selected rows and columns.

        Parameters
        ----------
        rows: List[int]
            Position indices of the rows.
        cols: List[int]
            Position indices of the cols.

        Returns
        -------
        ndarray
            A sub-DataFrame with selected rows and columns by position indices.
        """
        return self.data[array(rows)[:, None], array(cols)]

    def read_csv(self, filename: str, sep: str = ',', col_names: List[str] = None):
        """
        Read data from a .CSV file and save data to the caller object.

        Parameters
        ----------
        filename:str
            The path of the csv to be read.
        sep:str, default = ','
            Delimiter to use.
        col_names:List[str], optional
        """
        self.col_names = genfromtxt(filename, delimiter=sep, dtype=str, encoding='utf-8')[0]
        self.data = genfromtxt(filename, delimiter=sep, dtype=str, encoding='utf-8')[1:]
        if col_names is not None:
            if len(col_names) != len(self.data):
                raise Exception('Columns names length mismatch')
            else:
                self.col_names = asarray(col_names)
                pass
            pass
        pass

    def head(self, n: int = 5) -> ndarray:
        """
        Return the first `n` rows.

        Parameters
        ----------
        n:int, default = 5
            A positive integer which specifies the number of rows to select.

        Returns
        -------
        int
            The first `n` rows of the caller object
        """
        return self.data[0:n]
        pass

    def shape(self):
        """
        Return the shape of the caller object.
        Returns
        -------
        tuple[int]
            A tuple with lengths of each axis of the caller object.
        """
        return self.data.shape

    def train_test_split(
            self,
            test_size: float = 0.2,
            random_state: int = None,
            shuffle: bool = True
    ):
        """
        Shuffle and split the dataframe.

        Parameters
        ----------
        shuffle : bool, default = True
            Specify whether or not shuffle before splitting.
        test_size : float, default = 0.2
            The size of the testing set after splitting.
        random_state : int, optional
            Specify to reproduce same result when Running many times.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            The split training set and testing set.
        """
        index_list = arange(len(self.data))
        if shuffle:
            if random_state is not None:
                random.default_rng(random_state).shuffle(index_list)
            else:
                random.default_rng().shuffle(index_list)

        border = int(len(index_list) * (1 - test_size))
        train_indices = index_list[:border]
        test_indices = index_list[border:]

        return DataFrame(self.get_rows([train_indices])), DataFrame(self.get_rows([test_indices]))
