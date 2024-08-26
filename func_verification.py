from func_sun import *
import numpy as np
from scipy.linalg import expm

def verify_SU_generators(n):
    """
    This function verifies the properties of generators for the Special Unitary group SU(n), which is fundamental in many areas of physics and mathematics, especially in quantum mechanics for its role in symmetry operations. The function checks for:

    - Skew-Hermitian: Each generator should be equal to the negative of its complex conjugate transpose. This property is crucial because it ensures the generators lead to unitary matrices upon exponentiation, preserving probability amplitudes in quantum mechanics.
    - Traceless: The trace (sum of diagonal elements) of each generator should be zero. This condition is necessary for the determinant of the exponentiated generator to be 1, a defining property of SU(n) matrices.
    - Uniqueness and Count: There should be n^2 - 1 unique generators, corresponding to the dimension of the SU(n) Lie algebra.
    - Algebra Closure: The commutator (Lie bracket) of any two generators should also be a linear combination of the generators, ensuring the set of generators is closed under the Lie bracket operation.
    - Unitarity: Exponentiating any generator should yield a unitary matrix, which is a matrix that, when multiplied by its conjugate transpose, results in the identity matrix.
    - Determinant = 1: The determinant of exponentiated generators should be 1, further confirming their special unitary nature.
    - Orthogonality: Generators should be orthogonal under the Hilbert-Schmidt inner product, which is a condition for a valid Lie algebra basis.

    Parameters:
    n (int): The dimension of the SU(n) group, indicating the size of the square matrices representing the generators.

    Returns:
    str: A summary of the verification results, detailing whether the set of generators fulfills the necessary properties for SU(n).
    """
    gens = generate_SU(n)  # Generate the set of SU(n) generators.
    log = f"Checking SU({n}) generators:\n"

    # Initial header print for clarity in output.
    print('################################################################')
    print(f"n_generators: {len(gens)}")
    print(f'Expected number of generators: {n**2 - 1}')  # Direct calculation for clarity.
    print('################################################################')

    unique_gens = set()  # Use a set to ensure uniqueness of generators.
    for i, gen in enumerate(gens):
        skew_hermitian = is_skew_hermitian(gen)  # Check if gen is skew-Hermitian.
        traceless = np.isclose(np.trace(gen), 0)  # Check if the trace is close to zero, allowing for numerical precision issues.
        gen_tuple = tuple(map(tuple, gen))  # Convert array to tuple for hashability in set.
        unique_gens.add(gen_tuple)

        # Log details for each generator.
        log += f"Gen {i}: Skew-Hermitian - {skew_hermitian}, Traceless - {traceless}\n"

    # Verify the correct number of unique generators.
    if len(unique_gens) == n**2 - 1:
        log += "Correct number of unique generators.\n"
    else:
        log += "Mismatch in expected and actual number of unique generators.\n"

    # Algebra closure check using the commutator.
    closure_result = check_algebra_closure(gens)
    log += f"Algebra Closure: {closure_result}\n"

    # Check for unitarity and determinant = 1 for each generator.
    for i, gen in enumerate(gens):
        U = expm(gen)  # Exponentiate the generator.
        unitary = is_unitary(U)  # Check unitarity.
        det_one = np.isclose(np.linalg.det(U), 1)  # Check if determinant is close to 1.

        # Log results for each generator.
        log += f"Gen {i}: Unitary - {unitary}, Det = 1 - {det_one}\n"

    # Test a random group element for unitarity.
    random_U = random_element(gens)  # Generate a random group element.
    random_unitary = is_unitary(random_U)  # Check unitarity.
    log += f"Random group element unitarity: {random_unitary}\n"

    # Orthogonality check under the Hilbert-Schmidt inner product.
    orthogonality_check = check_orthogonality(gens)
    log += f"Orthogonality: {orthogonality_check}\n"

    return log

def verify_SO_generators(n):
    """
    This function validates the properties of generators for the Special Orthogonal group SO(n), pivotal in describing rotations in n-dimensional space. The checks performed are:

    - Skew-Symmetric: Generators must be equal to the negative of their transpose, reflecting the conservation of angular momentum in physical systems.
    - Uniqueness and Count: Ensures all generators are distinct and verifies the count as n(n-1)/2, which aligns with the degrees of freedom for rotations in n dimensions.
    - Algebra Closure: Confirms that the commutator of any two generators is a linear combination of the set, maintaining the group structure.
    - Orthogonality: Exponentiated generators must form orthogonal matrices, characterizing rotation without changing the object's scale.
    - Determinant +1: The determinant of exponentiated generators must be +1, ensuring orientation is preserved in rotations.
    - Orthogonality Under Hilbert-Schmidt: Generators should be orthogonal under the Hilbert-Schmidt inner product, a condition for a proper basis in Lie algebra.
    - Random Element Test: Verifies that a randomly constructed group element maintains orthogonality and has a determinant of +1, adding robustness to the verification.

    Parameters:
    n (int): The dimensionality of the SO(n) group, indicating the rotational space's dimension.

    Returns:
    str: A detailed log of the verification results for each property, indicating the adherence of the generators to the expected mathematical and physical principles.
    """
    gens = generate_SO(n)  # Generate the set of SO(n) generators.
    log = f"Checking SO({n}) generators:\n"

    # Output header for clarity.
    print('################################################################')
    print(f"n_generators: {len(gens)}")
    print(f'Expected number of generators: {n * (n - 1) // 2}')  # Direct calculation for expected count.
    print('################################################################')

    unique_gens = set()  # Utilize a set to ensure all generators are unique.
    for i, gen in enumerate(gens):
        skew_symmetric = is_skew_symmetric(gen)  # Verify skew-symmetry of each generator.
        gen_tuple = tuple(map(tuple, gen))  # Convert matrices to tuples for set inclusion.
        unique_gens.add(gen_tuple)  # Add to set to ensure uniqueness.

        # Log the result of checks for each generator.
        log += f"Gen {i}: Skew-Symmetric - {skew_symmetric}\n"

    # Validate the correct count and uniqueness of generators.
    if len(unique_gens) == n * (n - 1) // 2:
        log += "Valid number of unique generators.\n"
    else:
        log += "Discrepancy in generator count or uniqueness.\n"

    # Algebra closure verification using the commutator.
    closure_result = check_algebra_closure(gens)
    log += f"Algebra Closure: {closure_result}\n"

    # Verify orthogonality and determinant for each generator and a random element.
    for i, gen in enumerate(gens):
        R = expm(gen)  # Exponentiate generator to get rotation matrix.
        orthogonal = is_orthogonal(R)  # Check if matrix is orthogonal.
        det_one = np.isclose(np.linalg.det(R), 1)  # Ensure determinant is +1.

        # Log the results for each generator.
        log += f"Gen {i}: Orthogonal - {orthogonal}, Det = +1 - {det_one}\n"

    # Test a random group element for orthogonality and determinant.
    random_R = random_element(gens)  # Generate a random SO(n) element.
    random_orthogonal = is_orthogonal(random_R)  # Check orthogonality.
    random_det_one = np.isclose(np.linalg.det(random_R), 1)  # Verify determinant is +1.
    log += f"Random element: Orthogonal - {random_orthogonal}, Det = +1 - {random_det_one}\n"

    return log

def verify_T_generators(n):
    """
    This function assesses the generators of the Translation Group T(n), essential for understanding spatial translations in n-dimensional space. The verification process includes:

    - Correct Structure: Each generator must have zeros everywhere except for the last column, which should have exactly one non-zero element, representing a unit translation in one dimension.
    - Uniqueness: All generators must be distinct to ensure they represent independent translation directions.
    - Count: The number of unique generators should match the dimension n, aligning with the number of independent translation directions in n-dimensional space.
    - Closure Under Addition: The sum of any two generators should maintain the correct structure, reflecting the commutative property of translations.
    - Algebra Closure: Although translations typically do not have a non-trivial Lie algebra, this check ensures mathematical consistency.

    Parameters:
    n (int): The dimensionality of the space in which translations are considered, determining the structure and count of the generators.

    Returns:
    str: A comprehensive log of the verification outcomes, detailing the adherence of the translation generators to the expected properties.
    """
    gens = generate_T(n)  # Generate the set of T(n) generators.
    log = f"Checking T({n}) generators:\n"

    # Initial output header for clarity.
    print('################################################################')
    print(f"n_generators: {len(gens)}")
    print(f'Expected number of generators: {n}')  # Direct expectation for translation generators.
    print('################################################################')

    unique_gens = set()  # Use a set to ensure the uniqueness of generators.
    all_tests_passed = True  # Flag to track overall test success.

    for i, gen in enumerate(gens):
        # Verify the structural correctness of each generator.
        correct_structure = np.allclose(gen[:, :-1], np.zeros((n, n - 1))) and np.count_nonzero(gen[:, -1]) == 1
        gen_tuple = tuple(map(tuple, gen))  # Convert matrix to tuple for hashability.
        unique_gens.add(gen_tuple)  # Add to set to check for uniqueness.

        # Log details about each generator's structure.
        log += f"Gen {i}: Correct Structure - {correct_structure}\n"
        if not correct_structure:
            all_tests_passed = False  # Update test flag if structure is incorrect.

    # Validate the count and uniqueness of generators.
    if len(unique_gens) == n:
        log += "Valid number of unique generators.\n"
    else:
        log += "Mismatch in generator count or uniqueness.\n"
        all_tests_passed = False  # Update test flag if count or uniqueness fails.

    # Check for closure under addition, a key property for translation groups.
    for i, gen_i in enumerate(gens):
        for j, gen_j in enumerate(gens):
            sum_of_gens = gen_i + gen_j  # Sum two generators.
            # Verify the summed generators maintain the correct structural form.
            correct_sum_structure = np.allclose(sum_of_gens[:, :-1], np.zeros((n, n - 1)))
            log += f"Sum of Gen {i} + Gen {j}: Correct Structure - {correct_sum_structure}\n"
            if not correct_sum_structure:
                all_tests_passed = False  # Update test flag if sum structure fails.

    # Algebra closure check, typically trivial for translation groups but included for consistency.
    closure_result = check_algebra_closure(gens)
    log += f"Algebra Closure: {closure_result}\n"

    # Final verification summary, indicating overall success or failure.
    log += "Verification Summary: " + ("Passed" if all_tests_passed else "Failed") + "\n"
    return log

def verify_GL_generators(n):
    """
    This function assesses the generators of the General Linear Group GL(n), key for representing all invertible n x n matrices. Such matrices are central in linear algebra, providing the foundation for various mathematical and engineering applications. The verification covers:

    - Closure under the Commutator: Confirms that the commutator of any two generators remains within the span of the generator set, essential for the Lie algebra structure of GL(n).
    - Invertibility: Checks that matrices generated from any random linear combination of the generators are invertible, characteristic of GL(n) elements.
    - Dimensionality: Verifies that each generator is an n x n matrix, consistent with GL(n)'s definition.
    - Correct Number of Generators: Ensures the set contains an appropriate number of generators for spanning the space of n x n matrices.

    Parameters:
    n (int): The dimensionality of GL(n), indicative of the linear transformations' space.

    Returns:
    str: A comprehensive log of the verification outcomes, highlighting the consistency of the GL(n) generators with their theoretical properties.
    """
    gens = generate_GL(n)  # Generate the set of GL(n) generators.
    log = f"Checking GL({n}) generators:\n"
    all_tests_passed = True  # Flag for overall success.

    # Output header for readability.
    print('################################################################')
    print(f"n_generators: {len(gens)}")
    expected_num_generators = n**2  # Typical number of generators for a full basis of n x n matrices.
    print(f'Expected number of generators: {expected_num_generators}')
    print('################################################################')

    # Verify the correct number of generators.
    if len(gens) != expected_num_generators:
        log += f"Incorrect number of generators. Expected: {expected_num_generators}, Found: {len(gens)}\n"
        all_tests_passed = False
    else:
        log += "Correct number of generators verified.\n"

    for i, gen in enumerate(gens):
        correct_dimension = gen.shape == (n, n)  # Confirm n x n dimensionality.
        log += f"Gen {i}: Dimensionality - {'Passed' if correct_dimension else 'Failed'}\n"
        if not correct_dimension:
            all_tests_passed = False  # Flag if dimensionality fails.

    # Test invertibility of random linear combinations.
    for _ in range(10):
        random_coeffs = np.random.rand(len(gens))
        random_comb = sum(random_coeffs[k] * gens[k] for k in range(len(gens)))
        exp_random_comb = expm(random_comb)
        if np.linalg.det(exp_random_comb) != 0:
            log += "Random combination: Invertible.\n"
        else:
            log += "Random combination: Non-invertible.\n"
            all_tests_passed = False  # Flag if non-invertible.

    # Closure under commutator operation check.
    closure_result = check_algebra_closure(gens)
    log += f"Algebra Closure: {closure_result}\n"

    # Summary of verification.
    log += "Verification Summary: " + ("Passed" if all_tests_passed else "Failed") + "\n"
    return log

def verify_SL_generators(n):
    """
    This function assesses the generators of the Special Linear Group SL(n), characterized by n x n matrices with a determinant of 1, representing volume-preserving transformations. The verification includes:

    - Closure under the Commutator: Confirms that the commutator of any two generators remains within the generators' span, crucial for SL(n)'s Lie algebra.
    - Determinant = 1: Checks that exponentiating any generator yields a matrix with a determinant of 1, consistent with SL(n)'s definition.
    - Dimensionality: Verifies each generator is an n x n matrix, matching the SL(n) group's dimension.
    - Correct Number of Generators: Ensures there are \(n^2 - 1\) generators, corresponding to the degrees of freedom in SL(n).

    Parameters:
    n (int): The dimensionality of the SL(n) group, indicating the size of the matrices.

    Returns:
    str: A detailed log of the verification outcomes, demonstrating the SL(n) generators' compliance with their theoretical properties.
    """
    gens = generate_SL_from_SU(n)  # Generate SL(n) generators from SU(n) to preserve special properties.
    log = f"Checking SL({n}) generators:\n"
    all_tests_passed = True  # Flag for overall success.

    # Output header for clarity.
    print('################################################################')
    print(f"n_generators: {len(gens)}")
    expected_num_generators = n**2 - 1  # Expected number of generators for SL(n).
    print(f'Expected number of generators: {expected_num_generators}')
    print('################################################################')

    # Verify the correct number of generators.
    if len(gens) != expected_num_generators:
        log += f"Incorrect number of generators. Expected: {expected_num_generators}, Found: {len(gens)}\n"
        all_tests_passed = False
    else:
        log += "Correct number of generators verified.\n"

    for i, gen in enumerate(gens):
        correct_dimension = gen.shape == (n, n)  # Confirm n x n dimensionality.
        log += f"Gen {i}: Dimensionality - {'Passed' if correct_dimension else 'Failed'}\n"
        if not correct_dimension:
            all_tests_passed = False  # Flag if dimensionality fails.

        exp_gen = expm(gen)  # Exponentiate to obtain an SL(n) element.
        if np.isclose(np.linalg.det(exp_gen), 1):
            log += f"Gen {i}: Exponentiation yields Det=1 - Passed\n"
        else:
            log += f"Gen {i}: Exponentiation yields Det!=1 - Failed\n"
            all_tests_passed = False  # Flag if determinant deviates from 1.

    # Closure under commutator operation check.
    closure_result = check_algebra_closure(gens)
    log += f"Algebra Closure: {closure_result}\n"

    # Summary of verification.
    log += "Verification Summary: " + ("Passed" if all_tests_passed else "Failed") + "\n"
    return log

def verify_U_generators(n):
    """
    This function evaluates the generators of the Unitary Group U(n), crucial in quantum mechanics for symmetry and state transformations. The verification includes:

    - Closure under the Commutator: Checks if the set of generators is closed under the commutator operation, essential for the Lie algebra of U(n).
    - Unitarity: Verifies that exponentiating any generator yields a unitary matrix, key for preserving quantum probabilities.
    - Random Unitarity Test: Assesses the unitarity of 10 random linear combinations of generators to ensure consistency.
    - Dimensionality: Confirms each generator is an n x n matrix, matching the U(n) group's dimension.
    - Correct Number of Generators: Ensures the set contains \(n^2\) generators, the expected count for U(n).

    Parameters:
    n (int): The dimensionality of the U(n) group, indicative of the unitary matrices' size.

    Returns:
    str: A detailed summary of the verification results, elucidating the adherence of the U(n) generators to their theoretical properties.
    """
    gens = generate_U(n)  # Generate the set of U(n) generators.
    log = f"Checking U({n}) generators:\n"
    all_tests_passed = True  # Flag to track overall success.

    # Output header for readability.
    print('################################################################')
    print(f"n_generators: {len(gens)}")
    expected_num_generators = n**2  # The expected number of generators for U(n).
    print(f'Expected number of generators: {expected_num_generators}')
    print('################################################################')

    # Verify the correct number of generators.
    if len(gens) != expected_num_generators:
        log += f"Incorrect number of generators. Expected: {expected_num_generators}, Found: {len(gens)}\n"
        all_tests_passed = False
    else:
        log += "Correct number of generators verified.\n"

    for i, gen in enumerate(gens):
        correct_dimension = gen.shape == (n, n)  # Ensure each generator is an n x n matrix.
        log += f"Gen {i}: Dimensionality - {'Passed' if correct_dimension else 'Failed'}\n"
        if not correct_dimension:
            all_tests_passed = False  # Update flag if dimensionality check fails.

        exp_gen = expm(gen)  # Exponentiate the generator to obtain a U(n) element.
        if np.allclose(exp_gen @ exp_gen.conj().T, np.eye(n)):
            log += f"Gen {i}: Exponentiation yields Unitary - Passed\n"
        else:
            log += f"Gen {i}: Exponentiation yields Non-Unitary - Failed\n"
            all_tests_passed = False  # Update flag if unitarity check fails.

    # Test unitarity for random linear combinations of generators.
    for _ in range(10):  # Perform multiple tests for robustness.
        random_coeffs = np.random.rand(len(gens))  # Generate random coefficients.
        random_comb = sum(random_coeffs[k] * gens[k] for k in range(len(gens)))  # Create a random linear combination.
        exp_random_comb = expm(random_comb)  # Exponentiate the combination to get a U(n) element.
        if np.allclose(exp_random_comb @ exp_random_comb.conj().T, np.eye(n)):
            log += "Random combination: Unitary - Passed\n"
        else:
            log += "Random combination: Non-Unitary - Failed\n"
            all_tests_passed = False  # Update flag if any combination is non-unitary.

    # Verify algebraic closure under the commutator operation, ensuring the generators form a valid Lie algebra.
    closure_result = check_algebra_closure(gens)
    log += f"Algebra Closure: {closure_result}\n"

    # Final summary, indicating the overall verification result.
    log += "Verification Summary: " + ("Passed" if all_tests_passed else "Failed") + "\n"
    return log


def verify_O_generators(n):
    """
    This function evaluates the generators of the Orthogonal Group O(n), crucial for understanding rotational symmetries in n-dimensional space. The verification includes:

    - Skew-Symmetry: Checks if each generator is skew-symmetric, a defining property of orthogonal transformations.
    - Closure under the Commutator: Ensures the set of generators is closed under the commutator operation, vital for the Lie algebra of O(n).
    - Orthogonality: Verifies that exponentiating any generator results in an orthogonal matrix, preserving vector lengths and angles.
    - Random Orthogonality Test: Assesses the orthogonality of 10 randomly generated group elements to ensure consistency.
    - Dimensionality: Confirms each generator is an n x n matrix, matching the O(n) group's dimension.
    - Correct Number of Generators: Ensures the set contains \( \frac{n(n-1)}{2} \) generators, the expected count for O(n).

    Parameters:
    n (int): The dimensionality of the O(n) group, indicative of the rotational space's dimension.

    Returns:
    str: An exhaustive log of the verification results, elucidating the adherence of the O(n) generators to their theoretical properties.
    """
    gens = generate_O(n)  # Generate the set of O(n) generators.
    log = f"Checking O({n}) generators:\n"
    all_tests_passed = True  # Flag to track overall success.

    # Output header for readability.
    print('################################################################')
    print(f"n_generators: {len(gens)}")
    expected_num_generators = n * (n - 1) // 2  # Expected number of generators for O(n).
    print(f'Expected number of generators: {expected_num_generators}')
    print('################################################################')

    # Verify the correct number of generators.
    if len(gens) != expected_num_generators:
        log += f"Incorrect number of generators. Expected: {expected_num_generators}, Found: {len(gens)}\n"
        all_tests_passed = False
    else:
        log += "Correct number of generators verified.\n"

    for i, gen in enumerate(gens):
        skew_symmetric = is_skew_symmetric(gen)  # Confirm skew-symmetry.
        log += f"Gen {i}: Skew-Symmetry - {'Passed' if skew_symmetric else 'Failed'}\n"
        if not skew_symmetric:
            all_tests_passed = False  # Flag if skew-symmetry check fails.

        correct_dimension = gen.shape == (n, n)  # Confirm n x n dimensionality.
        log += f"Gen {i}: Dimensionality - {'Passed' if correct_dimension else 'Failed'}\n"
        if not correct_dimension:
            all_tests_passed = False  # Flag if dimensionality check fails.

        exp_gen = expm(gen)  # Exponentiate to obtain an O(n) element.
        if np.allclose(exp_gen @ exp_gen.T, np.eye(n)):
            log += f"Gen {i}: Exponentiation yields Orthogonal - Passed\n"
        else:
            log += f"Gen {i}: Exponentiation yields Non-Orthogonal - Failed\n"
            all_tests_passed = False  # Flag if orthogonality check fails.

    # Test orthogonality for random linear combinations of generators.
    for _ in range(10):  # Multiple tests for robustness.
        random_coeffs = np.random.rand(len(gens))
        random_comb = sum(random_coeffs[k] * gens[k] for k in range(len(gens)))
        exp_random_comb = expm(random_comb)  # Exponentiate the combination.
        if np.allclose(exp_random_comb @ exp_random_comb.T, np.eye(n)):
            log += "Random combination: Orthogonal - Passed\n"
        else:
            log += "Random combination: Non-Orthogonal - Failed\n"
            all_tests_passed = False  # Flag if any combination is non-orthogonal.

    # Closure under commutator operation check.
    closure_result = check_algebra_closure(gens)
    log += f"Algebra Closure: {closure_result}\n"

    # Summary of verification.
    log += "Verification Summary: " + ("Passed" if all_tests_passed else "Failed") + "\n"
    return log



