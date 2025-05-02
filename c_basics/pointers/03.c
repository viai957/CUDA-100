#include <stdio.h>

int main() {
    int num = 10;
    float fnum = 3.14;
    void* vptr;

    vptr = &num;
    printf("Integer: %d\n", *(int*)vptr); // Output: 10
    // vptr is a memory address "&num" but it is stored as a void pointer
    // We can't dereference a void pointer, so we cast it to an integer pointer
    // Then we dereference it with the final asterisk "*" to get the value

    vptr = &fnum;
    printf("Float: %f\n", *(float*)vptr); // Output: 3.14
    // vptr is a memory address "&fnum" but it is stored as a void pointer
    
}

// void pointers are used when we don't know the data type of the pointer
// fun fact: malloc returns a void pointer but we see it as a pointer to a memory block
// malloc is a function that allocates memory on the heap
// heap is a region of memory that is used to store dynamic data
// dynamic data is data that is created at runtime
// static data is data that is created at compile time

