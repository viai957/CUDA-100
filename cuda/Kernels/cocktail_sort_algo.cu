#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include <time.h>

/*
 * Cocktail Shaker Sort Benchmark
 *
 * Purpose:
 *   This program implements and benchmarks the Cocktail Shaker Sort algorithm on the host (CPU).
 *   Cocktail Shaker Sort is a bidirectional variant of Bubble Sort that sorts the array in both
 *   directions in each pass, improving performance on some datasets. This program is designed to
 *   measure execution time, throughput (in millions of comparisons per second), and verify correctness.
 *
 * Parameters:
 *   - n: Size of the array to sort (default: 1<<20, can be set via command line argument)
 *
 * Benchmarking:
 *   - Initializes the array with random floats in [0, 1).
 *   - Performs a warm-up run.
 *   - Times the sort over several runs (default: 5), reporting the average time.
 *   - Estimates the number of comparisons performed.
 *   - Verifies that the output array is sorted.
 *   - Prints a detailed summary table with metrics.
 *
 * Usage:
 *   ./cocktail_sort [n]
 *     n: Optional, number of elements in the array (default: 1048576)
 */

// Cocktail Shaker Sort implementation (host/CPU)
void cocktailShakerSort(float *A, int n) {
    if (n <= 1) return;
    int start = 0;
    int end = n - 1;
    bool swapped = true;
    while (swapped) {
        swapped = false;
        // Forward pass
        for (int i = start; i < end; ++i) {
            if (A[i] > A[i + 1]) {
                float tmp = A[i];
                A[i] = A[i + 1];
                A[i + 1] = tmp;
                swapped = true;
            }
        }
        if (!swapped)
            break;
        swapped = false;
        --end;
        // Backward pass
        for (int i = end - 1; i >= start; --i) {
            if (A[i] > A[i + 1]) {
                float tmp = A[i];
                A[i] = A[i + 1];
                A[i + 1] = tmp;
                swapped = true;
            }
        }
        ++start;
    }
}

// Verify that the array is sorted in non-decreasing order
bool verify_sorted(const float *A, int n) {
    for (int i = 1; i < n; ++i) {
        if (A[i-1] > A[i]) return false;
    }
    return true;
}

// Utility: fill array with random floats in [0, 1)
void fill_random(float *A, int n) {
    for (int i = 0; i < n; ++i)
        A[i] = (float)rand() / (float)RAND_MAX;
}

// Estimate number of comparisons performed by Cocktail Shaker Sort (worst case)
// For each pass, the unsorted region shrinks by 1 at each end.
// Total comparisons: sum_{k=0}^{n/2-1} 2*(n-1-2k)
size_t estimate_comparisons(int n) {
    size_t total = 0;
    int left = 0, right = n-1;
    while (left < right) {
        total += (right - left); // forward pass
        total += (right - left); // backward pass
        ++left;
        --right;
    }
    return total;
}

int main(int argc, char **argv) {
    // Parse optional command-line argument for array size
    int n = 1 << 20; // Default: 1048576
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            printf("Invalid array size '%s'. Using default n = %d\n", argv[1], 1 << 20);
            n = 1 << 20;
        }
    }

    printf("Cocktail Shaker Sort Benchmark (Host/CPU)\n");
    printf("Array size: n = %d\n", n);

    // Allocate arrays
    float *h_A = (float*)malloc(n * sizeof(float));
    float *h_A_ref = (float*)malloc(n * sizeof(float));
    if (!h_A || !h_A_ref) {
        fprintf(stderr, "Host memory allocation failed for n = %d\n", n);
        if (h_A) free(h_A);
        if (h_A_ref) free(h_A_ref);
        return 1;
    }

    // Initialize with random floats
    srand((unsigned int)time(NULL));
    fill_random(h_A, n);
    for (int i = 0; i < n; ++i) h_A_ref[i] = h_A[i];

    // Warm-up run (not timed)
    cocktailShakerSort(h_A_ref, n);
    // Re-initialize for actual runs
    fill_random(h_A, n);

    // Benchmarking parameters
    const int num_runs = 5;
    double total_time = 0.0;
    double min_time = 1e30, max_time = 0.0;

    printf("Running Cocktail Shaker Sort for %d runs...\n", num_runs);
    for (int run = 0; run < num_runs; ++run) {
        // Copy fresh random data for each run
        fill_random(h_A, n);
        auto t0 = std::chrono::high_resolution_clock::now();
        cocktailShakerSort(h_A, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        double t_sec = dt.count();
        total_time += t_sec;
        if (t_sec < min_time) min_time = t_sec;
        if (t_sec > max_time) max_time = t_sec;
        printf("  Run %d: %.6f sec\n", run + 1, t_sec);
        // Optional: verify after each run
        if (!verify_sorted(h_A, n)) {
            printf("  [!] ERROR: Array not sorted after run %d!\n", run + 1);
        }
    }

    double avg_time = total_time / num_runs;
    size_t cmp_count = estimate_comparisons(n);
    double throughput = (double)cmp_count / avg_time / 1e6; // millions of comparisons/sec

    // Final correctness check
    if (!verify_sorted(h_A, n)) {
        printf("ERROR: Final output array is not sorted!\n");
        free(h_A); free(h_A_ref);
        return 2;
    }

    // Print metrics
    printf("\n===== Benchmark Results =====\n");
    printf("Array size (n):             %d\n", n);
    printf("Estimated comparisons:      %zu\n", cmp_count);
    printf("Average time (sec):         %.6f\n", avg_time);
    printf("Min time (sec):             %.6f\n", min_time);
    printf("Max time (sec):             %.6f\n", max_time);
    printf("Throughput:                 %.2f million comparisons/sec\n", throughput);

    printf("\n%-25s %-15s\n", "Metric", "Value");
    printf("%-25s %-15d\n", "Array size", n);
    printf("%-25s %-15zu\n", "Comparisons", cmp_count);
    printf("%-25s %-15.6f\n", "Avg Time (sec)", avg_time);
    printf("%-25s %-15.6f\n", "Min Time (sec)", min_time);
    printf("%-25s %-15.6f\n", "Max Time (sec)", max_time);
    printf("%-25s %-15.2f\n", "Throughput (M cmp/s)", throughput);
    printf("\nVerification: Array is sorted: %s\n", verify_sorted(h_A, n) ? "PASS" : "FAIL");

    free(h_A);
    free(h_A_ref);
    return 0;
}