# include <stdio.h>

int main(){
    int arr[] = {12, 24, 36, 48, 60};
    int* ptr = arr; // ptr points to thefirst element of arr (default in C)

    printf("Posiiton one: %d\n", *ptr); // output: 12

    for (int i = 0; i < 5; i++){
        printf("%d", *ptr);
        printf("%p\n", ptr);
        ptr++; // move to the next element
    }

    // 
}