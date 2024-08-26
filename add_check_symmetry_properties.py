"""
Author: Dr. techn. Sebastian Raubitzek MSc. BSc.; SBA Research, Complexity and Resilience Research Group
This Python program, developed by Sebastian Raubitzek on February 7th, 2024, is designed to validate the generators of several key Lie groups. These groups, which include SU(n), SO(n), GL(n), SL(n), U(n), T(n), and O(n), are fundamental in various mathematical and physical theories, particularly in areas such as quantum mechanics, classical mechanics, linear algebra, and geometry. The program systematically checks the essential properties of each group's generators across different dimensions (from 2 to 6 for each group) to ensure they adhere to their mathematical definitions and characteristics.

Key functionalities include:
- Verifying that SU(n) generators are skew-Hermitian, traceless, and follow the algebra closure under the Lie bracket, among other properties, making them crucial in describing symmetries in quantum systems.
- Ensuring SO(n) generators are skew-symmetric and satisfy the special orthogonal conditions, reflecting rotational symmetries in Euclidean space.
- Checking GL(n) generators for their adherence to general linear transformations, including invertibility and dimensionality, relevant in broader linear transformations.
- Validating SL(n) generators for special linear conditions, especially maintaining determinant one, important for volume-preserving transformations.
- Confirming U(n) generators meet unitary conditions, preserving inner product, significant in quantum mechanics for state transformations.
- Testing T(n) generators, likely representing translations, to verify they correctly represent shifts in space.
- Assessing O(n) generators for orthogonality, including rotations and reflections, key in understanding symmetries in spaces with an inner product.

Each function dedicated to a specific group performs a series of checks to confirm the mathematical integrity of the generators, such as their structure, uniqueness, and compliance with theoretical expectations. The results of these verifications are printed in a structured format, making it easy to review and analyze the properties of each group's generators across the specified dimensions. This tool is invaluable for researchers, educators, and students in fields that rely on group theory and its applications, providing a reliable method to explore and validate fundamental aspects of Lie groups.

Usage:
The program is structured to loop over a range of dimensions and apply the verification functions to each group, printing the outcomes for each dimension. Users can directly run the program to see the results or modify the dimension range as needed for specific investigations or educational purposes.
"""
#from func_verification_legacy import *
from func_verification import *

# Example usage
for n in range(2, 8):
    print(f"Checking SU({n}):")
    result = verify_SU_generators(n)
    print(result)
    print('\n\n\n#################################################')
    print('#################################################')
    print(f"Checking SO({n}):")
    result = verify_SO_generators(n)
    print(result)
    print('\n\n\n#################################################')
    print('#################################################')
    print(f"Checking GL({n}):")
    result = verify_GL_generators(n)
    print(result)
    print('\n\n\n#################################################')
    print('#################################################')
    print(f"Checking SL({n}):")
    result = verify_SL_generators(n)
    print(result)
    print('\n\n\n#################################################')
    print('#################################################')
    print(f"Checking U({n}):")
    result = verify_U_generators(n)
    print(result)
    print('\n\n\n#################################################')
    print('#################################################')
    print(f"Checking T({n}):")
    result = verify_T_generators(n)
    print(result)
    print('\n\n\n#################################################')
    print('#################################################')
    print(f"Checking O({n}):")
    result = verify_O_generators(n)
    print(result)
    print('\n\n\n#################################################')
    print('#################################################')
