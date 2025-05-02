#include <stdio.h> // Standard input/output header file (for printf)

// & "address of" operator
// * "dereference" operator

int main() {
    int x = 10;
    int* ptr = &x; // & is used to get the memory address of a variable
    printf("Address of x: %p\n", ptr); // Output: memory address 
    printf("Value of x: %d\n", *ptr); // Output: 10
    // * in the prev line is used to get the value at the memory address
    // the memory address stored in the ptr (dereference)

}

