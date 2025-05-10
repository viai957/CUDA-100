#include <stdio.h>

void array_2d(){
    // Create and initialize a 2D array with 3 rows and 2 columns
    int arr[3][2] = {{0,1}, {2,3},{4,5}};

    // Print each element's value
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 2; j++){
            printf("arr[%d][%d] = %d\n", i, j, arr[i][j]);
        }
        printf("\n");
    }
}

void array_3d(){
    // Create and Initialize the 3-dimensional array 
    int arr[2][3][4] = {
        {   // First layer
            {1, 2, 3, 4},    // First row
            {5, 6, 7, 8},    // Second row
            {9, 10, 11, 12}  // Third row
        },
        {   // Second layer
            {13, 14, 15, 16}, // First row
            {17, 18, 19, 20}, // Second row
            {21, 22, 23, 24}  // Third row
        }
    };

    // Loop through the depth
    for (int i = 0; i < 2; i++){
        printf("Layer %d:\n", i);
        // Loop through the rows of each depth
        for (int j = 0; j < 3; ++j){
            // Loop through the columns of each row
            for (int k = 0; k < 4; ++k){
                printf("arr[%d][%d][%d] = %d\n", i, j, k, arr[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() {
    printf("Demonstrating 2D Array:\n");
    printf("=====================\n");
    array_2d();
    
    printf("\nDemonstrating 3D Array:\n");
    printf("=====================\n");
    array_3d();
    
    return 0;
}

