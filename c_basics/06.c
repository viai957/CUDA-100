#include <stdio.h>
// matrix -> arr -> integers
// similar to 0.1c but with arrays

int main() {
    int arr1[] = {1, 2, 3, 4, 5};
    int arr2[] = {6, 7, 8, 9, 10};
    int* ptr1 = arr1;
    int* ptr2 = arr2;
    int* matrix[] = {ptr1, ptr2};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++){
            printf("%d", *matrix[i]++);
        }
        printf("\n");
    }
}