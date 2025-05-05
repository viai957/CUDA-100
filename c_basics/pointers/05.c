#include <stdio.h>

int main() {
    int arr[] = {12, 24, 36, 48, 60};

    int* ptr = arr; // ptr points to the first element of arr (default in C)

    printf("position one: %d\n", *ptr); // Output: 12

    for (int i = 0; i < 5; i++){
        printf("%d\n", *ptr);
        printf("%p\t", ptr);
        ptr++;
    }
    // Output:
    // position one: 12
    // 12 0x7ffee28245e0
    // 24 0x7ffee28245e4
    // 36 0x7ffee28245e8
    // 48 0x7ffee28245ec

    // notice that the address of each element is 4 bytes apart (size of int = 4 bytes * 8 bits/32 = int32) each time.
    // ptrs are 64 bits in size (8 bytes). 2**32 = 4,294,967,296 bytes = 4GB
    // arrays are layed out in memoery in a contiguous manner (one after the other rather than in a random order)
}