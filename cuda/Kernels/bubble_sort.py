array = [64, 34, 25, 12, 22, 11, 90, 5]

n = len(array)
for i in range(n-1):
    for j in range(n-i-1):
        if array[j] > array[j+1]:
            array[j], array[j+1] = array[j+1], array[j]
print("Sorted array:", array)
