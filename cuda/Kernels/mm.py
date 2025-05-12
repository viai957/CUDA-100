def matrix_multiply(A, B):
    # Get dim
    m = len(A) # no of rows in A 
    k = len(A[0]) # no of cols in A 
    n = len(B[0]) # no of cols in B

    # Init results matrix c with zeros
    C = [[0 for _ in range(n)] for _ in range(m)]

    # Matrix multiplications
    for i in range(m):
        for j in range (n):
            sum = 0.0
            for l in range(k):
                sum += A[i][l] * B[l][j] 
            C[i][j] = sum
    return C

# Example : 
A = [[1,2], [3,4], [5,6]]
B = [[1, 2, 3, 4], [5, 6, 7, 8]]

C = matrix_multiply(A, B)

print("Matrix A (3x2):")
for row in A:
    print(row)

print("\n Matrix B (2x4):")
for row in B:
    print(row)

print("\n Result Matrix C (3x4):")
for row in C:
    print(row)

# Detailed calculation for first element
print("\n Detailed Calculation for C[0][0] :")
i, j = 0, 0
sum = 0
for l in range(len(A[0])):
    print(f"A[{i}][{l}] * B[{l}][j] = {A[i][j]} * {B[l][j]} = {A[i][l]} * {B[l][j]} =  {A[i][l] * B[l][j]}")
    sum += A[i][l] * B[l][j]
print(f"Sum = {sum}")