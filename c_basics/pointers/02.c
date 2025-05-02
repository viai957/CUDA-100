#include <stdio.h>

int main() {
    int value = 10;
    int* ptr = &value;
    int** ptr2 = &ptr;
    int*** ptr3 = &ptr2;

    printf("Value: %d\n", ***ptr3); // Output: 10
}