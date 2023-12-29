import numpy as np


def lu_decomposition_partial_pivoting(matrix):
    n = len(matrix)
    P = np.eye(n)
    L = np.zeros((n, n))
    U = matrix.copy()

    for k in range(n - 1):
        max_index = np.argmax(abs(U[k:n, k])) + k

        if max_index != k:
            U[[k, max_index]] = U[[max_index, k]]
            P[[k, max_index]] = P[[max_index, k]]
            if k >= 1:
                L[[k, max_index], :k] = L[[max_index, k], :k]

        for i in range(k + 1, n):
            if U[k, k] == 0:
                return None, None, None  # Return None values if determinant is zero
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:n] -= L[i, k] * U[k, k:n]

    if U[n - 1, n - 1] == 0:
        return None, None, None  # Return None values if determinant is zero

    for i in range(n):
        L[i, i] = 1

    return P, L, U


def solve_lu_decomposition(P, L, U, b):
    y = np.linalg.solve(L, P.dot(b))
    x = np.linalg.solve(U, y)
    return x


def get_square_matrix_from_user():
    print("لطفا سایز ماتریس مربعی را وارد کنید: ")
    size = int(input())
    matrix = []
    print(f"Enter {size}x{size} matrix elements separated by spaces or newlines:")
    for i in range(size):
        row = list(map(float, input().split()))
        if len(row) != size:
            print("Invalid input. Please enter exactly", size, "numbers.")
            return get_square_matrix_from_user()
        matrix.append(row)
    return np.array(matrix)


# Get square matrix input from the user
user_matrix = get_square_matrix_from_user()

# Check if the input matrix is square
if user_matrix.shape[0] != user_matrix.shape[1]:
    print("The input matrix is not square. Please enter a square matrix.")
else:
    # Check determinant and perform LU decomposition if determinant is not zero
    P, L, U = lu_decomposition_partial_pivoting(user_matrix)

    if P is None:
        print("The determinant of the matrix is zero. LU decomposition cannot be performed.")
    else:
        print("\nPivot matrix (P):")
        print(P)
        print("\nLower triangular matrix (L):")
        print(L)
        print("\nUpper triangular matrix (U):")
        print(U)

        # Solve for x in Ax = b
        print("\nEnter the values of the constants in the equation Ax = b separated by spaces:")
        constants = list(map(float, input().split()))
        if len(constants) != user_matrix.shape[0]:
            print("Invalid number of constants provided.")
        else:
            b = np.array(constants)
            x = solve_lu_decomposition(P, L, U, b)
            print("\nThe solution for x in Ax = b:")
            print(x)