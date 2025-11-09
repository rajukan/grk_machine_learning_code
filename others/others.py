import numpy as np
from scipy.sparse import csr_matrix

# # Define a dense matrix
# dense_matrix = np.array([[0, 0, 3], [0, 5, 2], [0, 0, 0]])
# # Convert the dense matrix to the sparse matrix
# sparse_matrix = csr_matrix(dense_matrix)
# # Print the sparse matrix
# print(sparse_matrix)

def isUgly_corrected(n: int) -> bool:
    # 1. Ugly numbers are positive only

    # 2. Define the prime factors for Ugly Numbers
    factors = [2, 3, 5]

    # 3. Repeatedly divide n by its allowed prime factors
    for factor in factors:
        print(f" starting with {n=}")
        while n % factor == 0:
            print(f"{factor=} divides {n=}, dividing...")
            n //= factor
            print(f"Now {n=}")

    # 4. If n is an Ugly Number, it must be reduced to 1 after all divisions
    result = (n == 1)
    print(f"Final {n=}, isUgly={result}")
    return result


# Example tests
series=[6,14,5,16,90]
for num in series:
    print(f"Starting for {num=}")
    isUgly_corrected(num)
