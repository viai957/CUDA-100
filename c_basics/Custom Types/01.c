// size_t = size type for memory allocation (unsigned long -> uint64)
// size_t is an unsigned integer data type used to represent the size of object in bytes
// It is guaranteed to be big enough to contain the size of the biggest object the host system can handel

# include <stdio.h>
# include <stdio.h>

int main() {
    int arr[] = {1, 2, 3, 4, 5};

    // size_t
    size_t size = sizeof(arr) / sizeof(arr[0]); // to get the total len of the array
    printf("Size of arr: %zu\n", size); // Output: 5
    printf("Size of size_t: %zu\n", sizeof(size_t)); // Output: 8 (bytes) -> 64 bits which is memory safe
    printf("int size in bytes: %zu\n", sizeof(int)); // Output: 4 (bytes) or 32 bits
    // z -> size_t
    // u -> unsigned int
    // %zu -> size_t
    // src: https://en.wikipedia.org/wiki/C_data_types

    return 0;
}

