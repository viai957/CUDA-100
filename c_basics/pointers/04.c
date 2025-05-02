// NULL pointer Initilization and safe usage.

// Key points:
// 1. Initilize pointers to NULL when they don't yet point to valid data
// 2. Check pointers for NULL before using to avoid crashes
// 3. NULL checks allow grceful handling of uninilized or failed allocations

# include <stdio.h>
# include <stdlib.h>

int main(){
    // Initialize pointer to NULL
    int* ptr = NULL;
    printf("1. Initial ptr value: %p\n", (void*)ptr);

    // Check for NULL before using
    if (ptr == NULL){
        
    }
}