import numpy as np

def calc_spatialfrequency(dimension, precision=False):
    """
    This function returns a square matrix of size [dimension X dimension]
    with the spatial frequencies calculated as row^2 + column^2.
    The null frequency is in [dim-1,dim-1].

    Parameters:
    - dimension: The dimension of the square matrix.
    - precision: If True, use double precision; otherwise, use single precision.

    Returns:
    - matrix: A square matrix of size [dimension X dimension].
    """

    dtype = np.float64 if precision else np.float32
    half_dim = dimension // 2

    temp_matrix = np.zeros((half_dim + 1, half_dim + 1), dtype=dtype)
    matrix = np.zeros((dimension, dimension), dtype=dtype)

    x = np.arange(half_dim + 1, dtype=dtype)  # Make a row
    for i in range(half_dim + 1):
        temp_matrix[0, i] = x[i]**2 + i**2  # Generate 1/4 of matrix

    # Place temp_matrix in the top-right corner and then
    # fill the rest of the matrix by rotating temp_matrix appropriately
    matrix[half_dim:, half_dim:] = temp_matrix
    matrix[:half_dim, :half_dim] = np.rot90(temp_matrix, 2)[1:, 1:]
    matrix[:half_dim, half_dim:] = np.rot90(temp_matrix, 1)[1:, :]
    matrix[half_dim:, :half_dim] = np.rot90(temp_matrix, 3)[:, 1:]

    return matrix
