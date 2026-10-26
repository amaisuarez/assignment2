import java.util.Arrays;

/**
 * Dense n x n matrix multiplication kernels (sequential, float64).
 *
 * Storage for all three: contiguous row-major double[] (element (i,j) at i*n + j),
 * identical to Assignment 1.
 *
 * Summation order. For every output element C[i][j] the three kernels add the
 * products A[i][k]*B[k][j] for k = 0, 1, ..., n-1 in increasing order, starting
 * from 0.0. Only the order in which DIFFERENT output elements are visited changes.
 * Java does not reassociate floating-point additions or fuse them into FMA
 * (strict IEEE 754 since Java 17, JEP 306), so the three kernels return
 * bit-identical results. tests/validate.py checks this.
 */
final class Dense {

    private Dense() {}

    /**
     * Baseline: the i-j-k triple loop from Assignment 1, copied unchanged from
     * java/MatMul.java (repo amaisuarez/assignment1, commit 754ec23).
     * The innermost loop walks B down a column (stride n), so for large n
     * almost every access to B touches a different cache line.
     */
    static void baseline(double[] A, double[] B, double[] C, int n) {
        for (int i = 0; i < n; i++) {
            int row = i * n;
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++)
                    s += A[row + k] * B[k * n + j];
                C[row + j] = s;
            }
        }
    }

    /**
     * Optimisation 1, loop reordering (i-k-j). Same arithmetic, loops k and j
     * swapped: the innermost loop now reads a row of B and updates a row of C,
     * both with stride 1, and A[i][k] stays in a register. C must start at zero
     * because it is accumulated in place; the O(n^2) zeroing is inside the timer.
     */
    static void ikj(double[] A, double[] B, double[] C, int n) {
        Arrays.fill(C, 0.0);
        for (int i = 0; i < n; i++) {
            int ri = i * n;
            for (int k = 0; k < n; k++) {
                double a = A[ri + k];
                int rk = k * n;
                for (int j = 0; j < n; j++)
                    C[ri + j] += a * B[rk + j];
            }
        }
    }

    /**
     * Optimisation 2, cache blocking (tiling) on top of the i-k-j order.
     * The three loops are split into tiles of bs x bs elements so that one tile
     * of A, B and C (3 * 8 * bs^2 bytes; 96 KiB for bs = 64) is reused while it is
     * still in cache, instead of streaming all of B once per row of C.
     * Tile loops run ii -> kk -> jj, so for a fixed (i, j) the k index is still
     * visited in increasing order (same rounding as the baseline). Edge tiles
     * handle n that is not a multiple of bs.
     */
    static void blocked(double[] A, double[] B, double[] C, int n, int bs) {
        Arrays.fill(C, 0.0);
        for (int ii = 0; ii < n; ii += bs) {
            int iMax = Math.min(ii + bs, n);
            for (int kk = 0; kk < n; kk += bs) {
                int kMax = Math.min(kk + bs, n);
                for (int jj = 0; jj < n; jj += bs) {
                    int jMax = Math.min(jj + bs, n);
                    for (int i = ii; i < iMax; i++) {
                        int ri = i * n;
                        for (int k = kk; k < kMax; k++) {
                            double a = A[ri + k];
                            int rk = k * n;
                            for (int j = jj; j < jMax; j++)
                                C[ri + j] += a * B[rk + j];
                        }
                    }
                }
            }
        }
    }
}
