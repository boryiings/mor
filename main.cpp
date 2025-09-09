// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "include/fake_quantize.cuh"


using namespace std;

int main() {
    // Integer 16256 in bf16 is 1.0. This can be used to test E4M3.
    int bf16_one = 16256;
    // Integer 17024 in bf16 is 64.0. This can be used to test E5M2.
    int bf16_sixty_four = 17024; 

    // Initialize arrays A, B, and C.
    uint32_t input[100], output[100];

    // Populate arrays A and B.
    for (int i = 0; i < 10; ++i) {
	for (int j = 0; j < 10; ++j) {
	    input[i * 10 + j] = i * 10 + j + bf16_one;
	}
    }

    fake_quantize(input, output, 10, 10, 5, 5, 0.0001);

    // Print out result.
    for (int i = 0; i < 10; ++i) {
	for (int j = 0; j < 10; ++j) {
	    cout << output[i * 10 + j] << " ";
	}
	cout << endl;
    }

    return 0;
}
