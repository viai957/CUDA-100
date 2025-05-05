// exact same as 02.c
# include <iostream>

using namespace std;

typedef struct {
    float x;
    float y;
} Point;

int main() {
    Point p = {1.2, 2.5};
    cout << "size of Point: " << sizeof(Point) << " bytes" << endl; // Output: 8 (bytes) = 4 bytes (float x) + 4 bytes (float y)
    return 0;
}