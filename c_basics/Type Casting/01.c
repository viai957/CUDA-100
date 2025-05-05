#include <stdio.h>

int main() {
    // c-style type casting
    float f = 69.69;
    int i = (int)f;
    printf("%d\n", i); // Output: 69 (rounded down since decimal is truncated (deletes the .69 part))
    // to char
}