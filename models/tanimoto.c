/* tanimoto.c — Pairwise Tanimoto similarity / distance for binary fingerprint arrays.
 *
 * Compile with OpenMP (preferred):
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o tanimoto.so tanimoto.c
 * Compile without OpenMP (fallback):
 *   gcc -O3 -march=native -shared -fPIC -o tanimoto.so tanimoto.c
 *
 * For binary {0,1} fingerprints:
 *   Tanimoto(a, b) = dot(a,b) / (sum(a) + sum(b) - dot(a,b))
 *
 * The inner loop computes dot and sum_b in one pass over d elements,
 * reusing the precomputed sum_a for row i.  With -O3 the compiler
 * auto-vectorises the inner loop (SIMD float32 operations).
 */

#include <stdlib.h>

/* tanimoto_matrix
 *   A   : float32 row-major  n_a x d
 *   B   : float32 row-major  n_b x d
 *   out : float32 row-major  n_a x n_b  (caller allocates)
 */
void tanimoto_matrix(const float *A, const float *B, float *out,
                     int n_a, int n_b, int d)
{
    /* Precompute per-row sums for A */
    float *sum_a = (float *)malloc((size_t)n_a * sizeof(float));
    for (int i = 0; i < n_a; i++) {
        const float *a = A + (size_t)i * d;
        float s = 0.0f;
        for (int k = 0; k < d; k++) s += a[k];
        sum_a[i] = s;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int i = 0; i < n_a; i++) {
        const float *a  = A + (size_t)i * d;
        float        sa = sum_a[i];
        float       *row = out + (size_t)i * n_b;

        for (int j = 0; j < n_b; j++) {
            const float *b  = B + (size_t)j * d;
            float dot = 0.0f, sb = 0.0f;
            for (int k = 0; k < d; k++) {
                dot += a[k] * b[k];
                sb  += b[k];
            }
            float denom = sa + sb - dot;
            row[j] = (denom > 1e-9f) ? dot / denom : 0.0f;
        }
    }

    free(sum_a);
}

/* tanimoto_distance_matrix — Tanimoto distance = 1 - Tanimoto similarity */
void tanimoto_distance_matrix(const float *A, const float *B, float *out,
                               int n_a, int n_b, int d)
{
    tanimoto_matrix(A, B, out, n_a, n_b, d);
    int total = n_a * n_b;
    for (int i = 0; i < total; i++) out[i] = 1.0f - out[i];
}
