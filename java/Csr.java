import java.util.Arrays;

/**
 * Compressed Sparse Row (CSR) matrix, n x n, float64 values. Own implementation,
 * no sparse library is used.
 *
 *   rowPtr[n+1]  row i owns positions rowPtr[i] .. rowPtr[i+1]-1
 *   col[nnz]     column index of each stored entry
 *   val[nnz]     value of each stored entry
 *
 * Storage: 4*(n+1) + 4*nnz + 8*nnz bytes (versus 8*n^2 for a dense matrix),
 * so CSR uses less memory than dense only while density < 2/3 (ignoring rowPtr).
 */
final class Csr {
    final int n;
    final int[] rowPtr;
    final int[] col;
    final double[] val;

    Csr(int n, int[] rowPtr, int[] col, double[] val) {
        this.n = n;
        this.rowPtr = rowPtr;
        this.col = col;
        this.val = val;
    }

    int nnz() { return rowPtr[n]; }

    long storageBytes() { return 4L * (n + 1) + 12L * nnz(); }

    /**
     * Conversion COO -> CSR. The COO triplets must be sorted row-major (row, then
     * column) without duplicates, which the input generator guarantees and this
     * method checks. Cost O(n + nnz). The arrays are copied so the CSR matrix is
     * independent of the COO buffers.
     */
    static Csr fromCoo(int n, int[] r, int[] c, double[] v) {
        int nnz = v.length;
        int[] rowPtr = new int[n + 1];
        for (int p = 0; p < nnz; p++) {
            if (p > 0 && (r[p] < r[p - 1] || (r[p] == r[p - 1] && c[p] <= c[p - 1])))
                throw new IllegalArgumentException("COO input not sorted row-major / has duplicates at " + p);
            rowPtr[r[p] + 1]++;
        }
        for (int i = 0; i < n; i++) rowPtr[i + 1] += rowPtr[i];
        return new Csr(n, rowPtr, Arrays.copyOf(c, nnz), Arrays.copyOf(v, nnz));
    }

    /**
     * C = A * B with Gustavson's row-by-row algorithm, in the two-pass form used
     * by SciPy's csr_matmat (symbolic pass to size the output, then numeric pass).
     *
     * Row i of C is the sum over the stored entries a_ik of row i of A of
     * a_ik * (row k of B). A dense accumulator acc[0..n) and a marker array
     * (mark[j] == i means column j already appeared in row i) merge the partial
     * rows. Work is proportional to the number of products sum_k nnz(A[:,k]) *
     * nnz(B[k,:]), not to n^3.
     *
     * Rounding: for a fixed (i, j) the products are added in increasing k (columns
     * of row i of A are sorted), starting from 0.0, exactly like the dense kernels,
     * which only add extra +-0.0 terms. Values therefore match the dense result
     * bit for bit (validated).
     *
     * Output: column indices inside a row are in order of first appearance (not
     * sorted). Entries that cancel to exactly 0.0 stay stored (structural nnz).
     * All output and work arrays are allocated inside this method, i.e. inside the
     * timed region, because the output size is not known in advance.
     */
    static Csr multiply(Csr A, Csr B) {
        int n = A.n;
        if (B.n != n) throw new IllegalArgumentException("size mismatch");
        int[] mark = new int[n];
        Arrays.fill(mark, -1);
        int[] cRowPtr = new int[n + 1];

        // pass 1 (symbolic): number of distinct columns in each row of C
        long nnz = 0;
        for (int i = 0; i < n; i++) {
            int count = 0;
            for (int p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
                int k = A.col[p];
                for (int q = B.rowPtr[k]; q < B.rowPtr[k + 1]; q++) {
                    int j = B.col[q];
                    if (mark[j] != i) { mark[j] = i; count++; }
                }
            }
            nnz += count;
            if (nnz > Integer.MAX_VALUE) throw new IllegalStateException("output too large for int indices");
            cRowPtr[i + 1] = (int) nnz;
        }

        // pass 2 (numeric)
        int[] cCol = new int[(int) nnz];
        double[] cVal = new double[(int) nnz];
        double[] acc = new double[n];
        Arrays.fill(mark, -1);
        for (int i = 0; i < n; i++) {
            int head = cRowPtr[i];
            int len = 0;
            for (int p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
                int k = A.col[p];
                double a = A.val[p];
                for (int q = B.rowPtr[k]; q < B.rowPtr[k + 1]; q++) {
                    int j = B.col[q];
                    if (mark[j] != i) {
                        mark[j] = i;
                        cCol[head + len++] = j;
                        acc[j] = 0.0;          // same start value as the dense kernels
                    }
                    acc[j] += a * B.val[q];
                }
            }
            for (int t = head; t < head + len; t++) cVal[t] = acc[cCol[t]];
        }
        return new Csr(n, cRowPtr, cCol, cVal);
    }

    /** Write this matrix into a dense row-major array (used only for validation). */
    void toDense(double[] D) {
        Arrays.fill(D, 0.0);
        for (int i = 0; i < n; i++)
            for (int p = rowPtr[i]; p < rowPtr[i + 1]; p++)
                D[i * n + col[p]] = val[p];
    }
}
