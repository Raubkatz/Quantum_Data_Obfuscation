"""
Author: Dr. techn. Sebastian Raubitzek MSc. BSc.; SBA Research, Complexity and Resilience Research Group
This module defines the `SymmetryFeatureMaps` class and associated functions to generate and apply symmetry-based feature maps to data points, leveraging various mathematical groups (e.g., SO, SU, SL, GL, etc.). The goal is to transform feature spaces by applying group-theoretic transformations and to add noise to these transformations in a controlled manner.

### Main Components:
1. **SymmetryFeatureMaps Class**:
   - This class handles the generation of group-based feature maps for a given number of features. The class supports a variety of groups, including SO, SL, SU, GL, U, O, Sp, and T.
   - The class includes methods for adding noise to these transformations, ensuring that the noise is not a linear combination of the group generators, thus maintaining the structural integrity of the transformations.

2. **Group Size Calculation Functions**:
   - Functions are provided to calculate the minimal group size required to accommodate the given number of features for different mathematical groups.
   - These include `find_su_group`, `find_sl_group`, `find_so_group`, `find_gl_u_group`, `find_sp_group`, `find_o_group`, and `find_translation_group`.

3. **Noise Addition**:
   - The class includes methods for adding noise to matrices, ensuring that the added noise cannot be represented as a linear combination of the group generators. This is particularly useful for creating robust transformations that maintain group-theoretic properties.

4. **Feature Map Application**:
   - The `apply_feature_map` method applies the appropriate group-based transformation to the input data, with options for adding noise, outputting real vectors, and handling different group types.

5. **Auxiliary Functions**:
   - Functions like `apply_group_element`, `random_element`, and `complex_to_real_matrix` support the main operations by applying transformations, generating random group elements, and converting complex matrices to real matrices, respectively.

### Key Classes and Functions:

- **SymmetryFeatureMaps**:
    - `__init__(self, num_features)`: Initializes the class with the number of features and generates the appropriate group generators.
    - `get_group_sizes(self)`: Returns a dictionary with the number of features and the size of each group.
    - `add_noise(self, matrix, noise_level=0.01)`: Adds uniform noise to a matrix.
    - `add_noise_non_generators(self, matrix, generators, noise_level=0.005)`: Adds noise that is not expressible as a linear combination of the group generators.
    - `add_noise_residual(self, matrix, noise_level=0.0005)`: Adds residual noise to a matrix.
    - `apply_feature_map(self, X, group_type, noise_level=0.0, output_real=False, return_group_n=False, multiply=0)`: Applies a group-based feature map to the input data.

- **Group Size Calculation Functions**:
    - `find_su_group(num_features)`: Determines the minimal size for the SU(n) group.
    - `find_sl_group(num_features)`: Determines the minimal size for the SL(n) group.
    - `find_so_group(num_features)`: Determines the minimal size for the SO(n) group.
    - `find_gl_u_group(num_features)`: Determines the minimal size for the GL or U group.
    - `find_sp_group(num_features)`: Determines the minimal size for the Sp group.
    - `find_o_group(num_features)`: Determines the minimal size for the O group.
    - `find_translation_group(num_features)`: Determines the minimal size for the translation group T(n).

- **Auxiliary Functions**:
    - `apply_group_element(data_point, generators)`: Applies a group element transformation to a data point.
    - `random_element(G)`: Generates a random element of the group represented by the list of generators.
    - `complex_to_real_matrix(complex_matrix)`: Converts a complex matrix to a real matrix by separating its real and imaginary components.

### Usage:
This module is designed to be integrated into machine learning pipelines where symmetry-based transformations of feature spaces are desired. It is particularly useful in scenarios involving complex, high-dimensional data where group-theoretic structures can be leveraged for data augmentation, feature extraction, or robustness testing.

### Dependencies:
- `numpy`: For numerical operations, particularly matrix manipulations.
- `scipy.linalg.expm`: For matrix exponential operations, essential for applying group transformations.
- `func_sun`: A custom module assumed to provide functions for generating group elements.
- `copy.deepcopy`: For creating deep copies of objects to avoid unintended side effects during transformations.
"""

import numpy as np
from scipy.linalg import expm
import func_sun
from copy import deepcopy as dc

class SymmetryFeatureMaps:
    def __init__(self, num_features):
        self.num_features = num_features
        # Initialize the group sizes
        size_SO = find_so_group(num_features)
        size_SL = find_sl_group(num_features)  # SL and SU have the same size
        size_SU = find_su_group(num_features)  # SL and SU have the same size
        size_GL = find_gl_u_group(num_features)
        size_U = find_gl_u_group(num_features)
        size_O = find_o_group(num_features)
        size_Sp = find_sp_group(num_features)
        size_T = find_translation_group(num_features)

        self.size_SO = size_SO
        self.size_SL = size_SL
        self.size_SU = size_SU
        self.size_GL = size_GL
        self.size_U = size_U
        self.size_O = size_O
        self.size_Sp = size_Sp
        self.size_T = size_T

        # Generate the corresponding generators for each group
        self.group_generators_SO = func_sun.generate_SO(size_SO)
        self.group_generators_SL = func_sun.generate_SL_from_SU(size_SL)
        self.group_generators_GL = func_sun.generate_GL(size_GL)
        self.group_generators_O = func_sun.generate_O(size_O)
        self.group_generators_U = func_sun.generate_U(size_U)
        self.group_generators_SU = func_sun.generate_SU(size_SU)
        self.group_generators_Sp = func_sun.generate_Sp(size_Sp)
        self.group_generators_T = func_sun.generate_T(size_T)

    def get_group_sizes(self):
        """
        Returns a dictionary containing the number of features and the number of generators for each group.
        """
        group_sizes = {
            "num_features": self.num_features,
            "size_SO": len(self.group_generators_SO),
            "size_SL": len(self.group_generators_SL),
            "size_SU": len(self.group_generators_SU),
            "size_GL": len(self.group_generators_GL),
            "size_U": len(self.group_generators_U),
            "size_O": len(self.group_generators_O),
            "size_Sp": len(self.group_generators_Sp),
            "size_T": len(self.group_generators_T)
        }
        return group_sizes

    def add_noise(self, matrix, noise_level=0.01):
        """
        Add noise to each component of the matrix.

        Parameters:
        matrix (np.ndarray): The input matrix.
        noise_level (float): The maximum magnitude of noise to add.

        Returns:
        np.ndarray: The matrix with added noise.
        """
        noise = np.random.uniform(-noise_level, noise_level, matrix.shape)
        if np.iscomplexobj(matrix):
            noise = noise + 1j * np.random.uniform(-noise_level, noise_level, matrix.shape)
        return matrix + noise

    def add_noise_non_generators(self, matrix, generators, noise_level=0.01):
        """
        Add noise to each component of the matrix.

        Parameters:
        matrix (np.ndarray): The input matrix.
        noise_level (float): The maximum magnitude of noise to add.

        Returns:
        np.ndarray: The matrix with added noise.
        """
        noise = np.random.uniform(-noise_level, noise_level, matrix.shape)
        if np.iscomplexobj(matrix):
            noise = noise + 1j * np.random.uniform(-noise_level, noise_level, matrix.shape)
        return matrix + noise

    def add_noise_non_generators(self, matrix, generators, noise_level=0.005):
        """
        Add noise to the matrix such that the noise component is not expressible as a linear combination of the generators.

        Parameters:
        matrix (np.ndarray): The input matrix.
        generators (list of np.ndarray): The list of generator matrices.
        noise_level (float): The maximum magnitude of noise to add.

        Returns:
        np.ndarray: The matrix with added residual noise.
        """

        def is_linear_combination(target, generators):
            """
            Check if the target matrix can be expressed as a linear combination of the generator matrices.
            """

            generator_stack = np.stack([gen.flatten() for gen in generators], axis=1) #flattenign the generators in order to simplify the linear combination stuff

            target_flat = target.flatten()
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(generator_stack, target_flat, rcond=None)
                residual_norm = np.linalg.norm(residuals)
                return residual_norm < 1e-10
            except np.linalg.LinAlgError:
                print('linear combinaion did not pẃork')
                return False

        noise = np.zeros_like(matrix)
        while True:
            noise = np.random.uniform(-noise_level, noise_level, matrix.shape)
            if np.iscomplexobj(matrix):
                noise = noise + 1j * np.random.uniform(-noise_level, noise_level, matrix.shape)
            if not is_linear_combination(noise, generators):
                break
        return matrix + noise

    def add_noise_residual(self, matrix, noise_level=0.0005):
        """
        Add noise to each component of the matrix.

        Parameters:
        matrix (np.ndarray): The input matrix.
        noise_level (float): The maximum magnitude of noise to add.

        Returns:
        np.ndarray: The matrix with added noise.
        """
        noise = np.random.uniform(-noise_level, noise_level, matrix.shape)
        if np.iscomplexobj(matrix):
            noise = noise + 1j * np.random.uniform(-noise_level, noise_level, matrix.shape)
        return matrix + noise

    def apply_feature_map(self, X, group_type, noise_level=0.0, output_real=False, return_group_n=False, multiply=0):
        if group_type == "SO":
            if return_group_n: return self.SOn_feature_map(X, noise_level, output_real=output_real, multiply=multiply), self.size_SO
            else: return self.SOn_feature_map(X, noise_level, output_real=output_real, multiply=multiply)
        elif group_type == "SL":
            if return_group_n: return self.SLn_feature_map(X, noise_level, output_real=output_real, multiply=multiply), self.size_SL
            else: return self.SLn_feature_map(X, noise_level, output_real=output_real, multiply=multiply)
        elif group_type == "SU":
            if return_group_n: return self.SUn_feature_map(X, noise_level, output_real=output_real, multiply=multiply), self.size_SU
            else: return self.SUn_feature_map(X, noise_level, output_real=output_real, multiply=multiply)
        elif group_type == "GL":
            if return_group_n: return self.GLn_feature_map(X, noise_level, output_real=output_real, multiply=multiply), self.size_GL
            else: return self.GLn_feature_map(X, noise_level, output_real=output_real, multiply=multiply)
        elif group_type == "U":
            if return_group_n: return self.Un_feature_map(X, noise_level, output_real=output_real, multiply=multiply), self.size_U
            else: return self.Un_feature_map(X, noise_level, output_real=output_real, multiply=multiply)
        elif group_type == "O":
            if return_group_n: return self.On_feature_map(X, noise_level, output_real=output_real, multiply=multiply), self.size_O
            else: return self.On_feature_map(X, noise_level, output_real=output_real, multiply=multiply)
        elif group_type == "Sp":
            if return_group_n: return self.Spn_feature_map(X, noise_level, output_real=output_real, multiply=multiply), self.size_Sp
            else: return self.Spn_feature_map(X, noise_level, output_real=output_real, multiply=multiply)
        elif group_type == "T":
            if return_group_n: return self.Tn_feature_map(X, noise_level, output_real=output_real, multiply=multiply), self.size_T
            else: return self.Tn_feature_map(X, noise_level, output_real=output_real, multiply=multiply)
        else:
            raise ValueError(f"Unknown group type: {group_type}")

    def SOn_feature_map(self, X, noise_level=0.0, output_real=False, multiply=0):
        if multiply == 0:
            return self.generic_feature_map(X, self.group_generators_SO, noise_level, output_real=output_real)
        else: return self.generic_feature_map_synth(X, self.group_generators_SO, noise_level, output_real=output_real, multiply=multiply)

    def SLn_feature_map(self, X, noise_level=0.0, output_real=False, multiply=0):
        if multiply == 0:
            return self.generic_feature_map(X, self.group_generators_SL, noise_level, output_real=output_real)
            #return self.generic_feature_map_synth(X, self.group_generators_T, noise_level, output_real=output_real,
            #                                  multiply=1)
        else: return self.generic_feature_map_synth(X, self.group_generators_SL, noise_level, output_real=output_real, multiply=multiply)

    def SUn_feature_map(self, X, noise_level=0.0, output_real=False, multiply=0):
        if multiply == 0:
            #return self.generic_feature_map_synth(X, self.group_generators_T, noise_level, output_real=output_real,
            #                                      multiply=1)
            return self.generic_feature_map(X, self.group_generators_SU, noise_level, output_real=output_real)
        else: return self.generic_feature_map_synth(X, self.group_generators_SU, noise_level, output_real=output_real, multiply=multiply)

    def GLn_feature_map(self, X, noise_level=0.0, output_real=False, multiply=0):
        if multiply == 0:
            return self.generic_feature_map(X, self.group_generators_GL, noise_level, output_real=output_real)
        else: return self.generic_feature_map_synth(X, self.group_generators_GL, noise_level, output_real=output_real, multiply=multiply)

    def Un_feature_map(self, X, noise_level=0.0, output_real=False, multiply=0):
        if multiply == 0:
            return self.generic_feature_map(X, self.group_generators_U, noise_level, output_real=output_real)
        else: return self.generic_feature_map_synth(X, self.group_generators_U, noise_level, output_real=output_real, multiply=multiply)

    def On_feature_map(self, X, noise_level=0.0, output_real=False, multiply=0):
        if multiply == 0:
            return self.generic_feature_map(X, self.group_generators_O, noise_level, output_real=output_real)
        else: return self.generic_feature_map_synth(X, self.group_generators_O, noise_level, output_real=output_real, multiply=multiply)

    def Spn_feature_map(self, X, noise_level=0.0, output_real=False, multiply=0):
        if multiply == 0:
            return self.generic_feature_map(X, self.group_generators_Sp, noise_level, output_real=output_real)
        else: return self.generic_feature_map_synth(X, self.group_generators_Sp, noise_level, output_real=output_real, multiply=multiply)

    def Tn_feature_map(self, X, noise_level=0.0, output_real=False, multiply=0):
        if multiply == 0:
            return self.generic_feature_map(X, self.group_generators_T, noise_level, output_real=output_real)
        else: return self.generic_feature_map_synth(X, self.group_generators_T, noise_level, output_real=output_real, multiply=multiply)

    def complex_to_real_vector(self, vector):
        """
        Convert a complex vector with n components into a real vector with 2n components.
        """
        real_part = vector.real
        imag_part = vector.imag

        return np.concatenate((real_part, imag_part))

    def generic_feature_map_old_working(self, X, generators, noise_level, output_real=False):
        num_features = len(X)
        num_generators = len(generators)

        group_element = np.sum([X[i] * generators[i] for i in range(min(num_features, num_generators))], axis=0)
        if noise_level != 0.0:
            group_element = self.add_noise(group_element, noise_level)
        group_element = expm(1j * group_element)

        dim = generators[0].shape[0]
        initial_vector = np.ones(dim) / np.sqrt(dim)

        transformed_vector = np.dot(group_element, initial_vector)

        # Convert to real vector if required
        if output_real:
            transformed_vector = self.complex_to_real_vector(transformed_vector)

        return transformed_vector

    def generic_feature_map(self, X, generators, noise_level, output_real=False):
        num_features = len(X)
        num_generators = len(generators)

        if noise_level != 0.0: #ne noise level added
            noise_vector = np.random.uniform(0, 1, num_features)
            noise_vector = dc(noise_vector / np.sum(noise_vector) * noise_level)
            X = dc(X + noise_vector)
            group_element = np.sum([X[i] * generators[i] for i in range(min(num_features, num_generators))], axis=0)
            group_element = self.add_noise_non_generators(group_element, generators, noise_level=0.0005)  # adding the residual noise, which must not be expressible as a linear combination of the generators
        else:
            group_element = np.sum([X[i] * generators[i] for i in range(min(num_features, num_generators))], axis=0)
        group_element = expm(1j * group_element)

        dim = generators[0].shape[0]
        initial_vector = np.ones(dim) / np.sqrt(dim)

        transformed_vector = np.dot(group_element, initial_vector)

        # Convert to real vector if required
        if output_real:
            transformed_vector = self.complex_to_real_vector(transformed_vector)

        return transformed_vector


    def generic_feature_map_synth_old_working(self, X, generators, noise_level, output_real=False, multiply=2):
        num_features = len(X)
        num_generators = len(generators)

        transformed_vectors = list()
        for i in range(multiply):
            group_element = np.sum([X[i] * generators[i] for i in range(min(num_features, num_generators))], axis=0)
            if noise_level != 0.0:
                group_element = self.add_noise(group_element, noise_level)
            group_element = expm(1j * group_element)

            dim = generators[0].shape[0]
            initial_vector = np.ones(dim) / np.sqrt(dim)

            transformed_vector = np.dot(group_element, initial_vector)
            # Convert to real vector if required
            if output_real:
                transformed_vector = self.complex_to_real_vector(transformed_vector)

            transformed_vectors.append(transformed_vector)

        return np.array(transformed_vectors)


    def generic_feature_map_synth(self, X, generators, noise_level, output_real=False, multiply=2):
        num_features = len(X)
        num_generators = len(generators)

        transformed_vectors = list()
        for i in range(multiply): #new noise level added

            if noise_level != 0.0:
                noise_vector = np.random.uniform(0, 1, num_features)
                noise_vector = dc(noise_vector / np.sum(noise_vector) * noise_level)
                X = dc(X + noise_vector)
                group_element = np.sum([X[i] * generators[i] for i in range(min(num_features, num_generators))], axis=0)
                group_element = self.add_noise_non_generators(group_element, generators, noise_level=0.0005) #adding the residual noise, which must not be expressible as a linear combination of the generators
            else:
                group_element = np.sum([X[i] * generators[i] for i in range(min(num_features, num_generators))], axis=0)
            group_element = expm(1j * group_element)

            dim = generators[0].shape[0]
            initial_vector = np.ones(dim) / np.sqrt(dim)

            transformed_vector = np.dot(group_element, initial_vector)
            # Convert to real vector if required
            if output_real:
                transformed_vector = self.complex_to_real_vector(transformed_vector)

            transformed_vectors.append(transformed_vector)

        return np.array(transformed_vectors)

def find_su_group(num_features):
    """
    Find the smallest n for SU(n) that provides enough generators for the number of features.
    """
    n = 2  # Start from SU(2)
    while True:
        num_generators = n**2 - 1
        if num_generators >= num_features:
            # Ensure at least equal number of generators as features
            return max(n, int(np.ceil(np.sqrt(num_features + 1))))
        n += 1

def find_sl_group(num_features):
    return find_su_group(num_features)  # Same as SU(n)


def find_so_group(num_features):
    n = 1  # Start from SO(1)
    while True:
        num_generators = n * (n - 1) // 2
        if num_generators >= num_features:
            return n
        n += 1

def find_gl_u_group(num_features):
    n = 1  # Start from GL(1) or U(1)
    while True:
        num_generators = n**2
        if num_generators >= num_features:
            return n
        n += 1

def find_sp_group(num_features):
    n = 1  # Start from Sp(1)
    while True:
        num_generators = n * (2 * n + 1)
        if num_generators >= num_features:
            return n
        n += 1

def find_o_group(num_features):
    return find_so_group(num_features)  # Same as SO(n)

def find_translation_group(num_features):
    """
    Find the smallest n for the translation group T(n) that provides enough generators for the number of features.
    In the case of the translation group, each dimension has a single generator.
    """
    n = 1  # Start from T(1)
    while True:
        num_generators = n  # Each dimension has one generator
        if num_generators >= num_features:
            return n
        n += 1


def apply_group_element(data_point, generators):
    """
    Apply the group element corresponding to the data point to the vector.
    """
    # Number of features and generators
    num_features = len(data_point)
    num_generators = len(generators)

    # Scale data point
    scaled_data_point = data_point

    # Create the group element
    group_element = np.sum([scaled_data_point[i] * generators[i] for i in range(min(num_features, num_generators))], axis=0)
    group_element = expm(1j * group_element)

    # Prepare the initial vector
    dim = generators[0].shape[0]
    initial_vector = np.ones(dim) / np.sqrt(dim)

    # Apply the group element
    transformed_vector = np.dot(group_element, initial_vector)

    return transformed_vector

def random_element(G):
    """
    Generate a random element of the group represented by the list of generators G.

    A random element is created by forming a linear combination of the generators
    with random coefficients, and then exponentiating the resulting matrix.

    Parameters
    ----------
    G : list of array-like
        List of generators of the group.

    Returns
    -------
    array-like
        A random element of the group.
    """
    # Get the dimension of the group from the first generator
    N = G[0].shape[0]
    # Get the number of generators
    num_generators = len(G)

    # Generate a set of random coefficients
    coefficients = np.random.rand(num_generators)

    # Create a linear combination of the generators with the coefficients
    linear_combination = sum(coefficients[i] * G[i] for i in range(num_generators))

    # Apply the matrix exponential operation
    R = expm(1j * linear_combination)

    return R

def complex_to_real_matrix(complex_matrix):
    """
    Convert a complex matrix to a real matrix.

    Parameters:
    complex_matrix (np.ndarray): A complex-valued matrix.

    Returns:
    np.ndarray: A real-valued matrix.
    """
    # Get the dimensions of the original complex matrix
    m, n = complex_matrix.shape

    # Initialize a real matrix with double the dimensions
    real_matrix = np.zeros((2 * m, 2 * n))

    # Fill the real matrix with the real and imaginary parts of the complex matrix
    for i in range(m):
        for j in range(n):
            real_matrix[2*i][2*j] = complex_matrix[i][j].real
            real_matrix[2*i][2*j+1] = -complex_matrix[i][j].imag
            real_matrix[2*i+1][2*j] = complex_matrix[i][j].imag
            real_matrix[2*i+1][2*j+1] = complex_matrix[i][j].real

    return real_matrix
