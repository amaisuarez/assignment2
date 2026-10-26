import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Benchmark driver for Assignment 2: one process = one (method, input) configuration.
 *
 * Methods: baseline | ikj | blocked  (dense, Part A and dense reference of Part B)
 *          sparse                     (CSR x CSR -> CSR, Part B)
 *
 * Input formats (both little-endian):
 *   dense  raw float64, row-major, n*n values, no header (same files as Assignment 1)
 *   coo    int64 n, int64 nnz, int32 rows[nnz], int32 cols[nnz], float64 vals[nnz],
 *          sorted row-major, no duplicates
 * Dense methods may read COO files (densified before timing). The sparse method
 * reads COO only, so its process never holds an n x n dense array.
 *
 * Sequence inside the process (only step 4 produces the reported samples):
 *   1. load inputs; for sparse also build the CSR matrices          (not timed)
 *   2. W warm-up rounds of >= target time each                      (timed, reported as "warmup")
 *   3. batch (only if --batch 0): t = time per multiplication in the last warm-up
 *      round (or one extra timed multiplication if W = 0),
 *      batch = max(1, ceil(target / t))                            (reported as "calibration")
 *   4. R measured repetitions, each times `batch` multiplications  (reported as "measured")
 *   5. sparse only: R timed COO->CSR conversions of A and B         (reported as "convert")
 *
 * Timer: System.nanoTime() (monotonic). Each sample line:
 *   kind,index,elapsed_ns,batch,checksum,nnz_out
 * The checksum (sum of all output values) consumes the result so the JIT cannot
 * remove the computation. Lines starting with '#' carry metadata.
 */
public final class MatMul2 {

    // ------------------------------------------------------------------ I/O
    static ByteBuffer readAll(String path) throws IOException {
        return ByteBuffer.wrap(Files.readAllBytes(Path.of(path))).order(ByteOrder.LITTLE_ENDIAN);
    }

    static double[] loadDense(String path, int n) throws IOException {
        ByteBuffer buf = readAll(path);
        if (buf.capacity() != 8L * n * n)
            throw new IOException(path + ": expected " + (long) n * n + " float64 values");
        double[] m = new double[n * n];
        buf.asDoubleBuffer().get(m);
        return m;
    }

    /** COO triplets as three parallel arrays. */
    record Coo(int n, int[] r, int[] c, double[] v) {
        long bytes() { return 16L * v.length; }

        double[] toDense() {
            double[] d = new double[n * n];
            for (int p = 0; p < v.length; p++) d[r[p] * n + c[p]] = v[p];
            return d;
        }
    }

    static Coo loadCoo(String path, int expectN) throws IOException {
        ByteBuffer buf = readAll(path);
        long n = buf.getLong(), nnz = buf.getLong();
        if (n != expectN) throw new IOException(path + ": file has n=" + n + ", expected " + expectN);
        if (buf.capacity() != 16 + 16 * nnz) throw new IOException(path + ": bad size for nnz=" + nnz);
        int[] r = new int[(int) nnz], c = new int[(int) nnz];
        double[] v = new double[(int) nnz];
        buf.asIntBuffer().get(r);
        buf.position(16 + 4 * (int) nnz);
        buf.slice().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(c);
        buf.position(16 + 8 * (int) nnz);
        buf.slice().order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(v);
        return new Coo((int) n, r, c, v);
    }

    static void saveDense(String path, double[] C) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * C.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.asDoubleBuffer().put(C);
        Files.write(Path.of(path), buf.array());
    }

    // -------------------------------------------------------------- helpers
    static double sum(double[] x) {
        double s = 0.0;
        for (double v : x) s += v;
        return s;
    }

    static int countNonzero(double[] x) {
        int c = 0;
        for (double v : x) if (v != 0.0) c++;
        return c;
    }

    /** Peak resident set size of this process (Linux VmHWM); -1 if unavailable. */
    static long peakRssBytes() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/self/status")))
                if (line.startsWith("VmHWM:"))
                    return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024;
        } catch (IOException | RuntimeException e) { /* not Linux */ }
        return -1;
    }

    // ------------------------------------------------------------ the runner
    /** One method bound to its inputs; run(k) performs k multiplications. */
    interface Kernel {
        void run(int times);
        double checksum();
        int nnzOut();
    }

    static final class DenseKernel implements Kernel {
        final String method;
        final double[] A, B, C;
        final int n, bs;

        DenseKernel(String method, double[] A, double[] B, int n, int bs) {
            if (!method.equals("baseline") && !method.equals("ikj") && !method.equals("blocked"))
                throw new IllegalArgumentException("unknown dense method " + method);
            this.method = method;
            this.A = A;
            this.B = B;
            this.C = new double[n * n];                     // allocated outside the timer
            this.n = n;
            this.bs = bs;
        }

        public void run(int times) {
            for (int t = 0; t < times; t++) {
                switch (method) {
                    case "baseline" -> Dense.baseline(A, B, C, n);
                    case "ikj" -> Dense.ikj(A, B, C, n);
                    default -> Dense.blocked(A, B, C, n, bs);
                }
            }
        }
        public double checksum() { return sum(C); }
        public int nnzOut() { return countNonzero(C); }
    }

    static final class SparseKernel implements Kernel {
        final Csr A, B;
        Csr C;
        SparseKernel(Csr A, Csr B) { this.A = A; this.B = B; }
        public void run(int times) { for (int t = 0; t < times; t++) C = Csr.multiply(A, B); }
        public double checksum() { return sum(C.val); }
        public int nnzOut() { return C.nnz(); }
    }

    public static void main(String[] args) throws IOException {
        String method = null, pa = null, pb = null, out = null, format = "dense";
        int n = -1, warmup = 0, reps = 1, batch = 0, bs = 64;
        double targetMs = 100;
        for (int i = 0; i + 1 < args.length; i += 2) {
            switch (args[i]) {
                case "--method" -> method = args[i + 1];
                case "--n" -> n = Integer.parseInt(args[i + 1]);
                case "--a" -> pa = args[i + 1];
                case "--b" -> pb = args[i + 1];
                case "--format" -> format = args[i + 1];
                case "--block" -> bs = Integer.parseInt(args[i + 1]);
                case "--warmup" -> warmup = Integer.parseInt(args[i + 1]);
                case "--reps" -> reps = Integer.parseInt(args[i + 1]);
                case "--batch" -> batch = Integer.parseInt(args[i + 1]);
                case "--target-ms" -> targetMs = Double.parseDouble(args[i + 1]);
                case "--out" -> out = args[i + 1];
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (method == null || n <= 0 || pa == null || pb == null) {
            System.err.println("usage: MatMul2 --method baseline|ikj|blocked|sparse --n N --a A --b B "
                    + "[--format dense|coo] [--block 64] [--warmup W] [--reps R] [--batch K|0=auto] "
                    + "[--target-ms 100] [--out C.bin]");
            System.exit(1);
        }
        if ((long) n * n > Integer.MAX_VALUE) throw new IllegalArgumentException("n too large for a flat double[]");
        if (bs <= 0) throw new IllegalArgumentException("block size must be positive");

        // ---- 1. load (not timed)
        Kernel kernel;
        Coo cooA = null, cooB = null;
        long storage;
        if (method.equals("sparse")) {
            if (!format.equals("coo")) throw new IllegalArgumentException("sparse method needs --format coo");
            cooA = loadCoo(pa, n);
            cooB = loadCoo(pb, n);
            Csr A = Csr.fromCoo(n, cooA.r(), cooA.c(), cooA.v());
            Csr B = Csr.fromCoo(n, cooB.r(), cooB.c(), cooB.v());
            System.out.println("#nnz_a," + A.nnz());
            System.out.println("#nnz_b," + B.nnz());
            System.out.println("#input_coo_bytes," + (cooA.bytes() + cooB.bytes()));
            kernel = new SparseKernel(A, B);
            storage = A.storageBytes() + B.storageBytes();      // C added after the run
        } else {
            double[] A, B;
            if (format.equals("coo")) {
                Coo ca = loadCoo(pa, n), cb = loadCoo(pb, n);
                System.out.println("#nnz_a," + ca.v().length);
                System.out.println("#nnz_b," + cb.v().length);
                A = ca.toDense();
                B = cb.toDense();
            } else {
                A = loadDense(pa, n);
                B = loadDense(pb, n);
                System.out.println("#nnz_a," + countNonzero(A));
                System.out.println("#nnz_b," + countNonzero(B));
            }
            kernel = new DenseKernel(method, A, B, n, bs);
            storage = 3L * 8 * n * n;
        }

        // ---- 2. warm-up: each warm-up round repeats the multiplication until at
        //         least targetMs have passed (at least once), so that small sizes
        //         also run long enough for the JIT to compile the kernel.
        double lastWarmupPerMult = -1;
        for (int r = 0; r < warmup; r++) {
            long t0 = System.nanoTime(), t1;
            int count = 0;
            do {
                kernel.run(1);
                count++;
                t1 = System.nanoTime();
            } while (t1 - t0 < targetMs * 1e6);
            lastWarmupPerMult = (double) (t1 - t0) / count;
            System.out.println("warmup," + r + "," + (t1 - t0) + "," + count + "," + kernel.checksum() + "," + kernel.nnzOut());
        }

        // ---- 3. batch size: from the last warm-up round if there was one,
        //         otherwise from one extra timed multiplication
        if (batch <= 0 && lastWarmupPerMult > 0) {
            batch = (int) Math.max(1, Math.ceil(targetMs * 1e6 / lastWarmupPerMult));
        } else if (batch <= 0) {
            long t0 = System.nanoTime();
            kernel.run(1);
            long t1 = System.nanoTime();
            long t = Math.max(1, t1 - t0);
            batch = (int) Math.max(1, Math.ceil(targetMs * 1e6 / t));
            System.out.println("calibration,0," + t + ",1," + kernel.checksum() + "," + kernel.nnzOut());
        }

        // ---- 4. measured repetitions
        for (int r = 0; r < reps; r++) {
            long t0 = System.nanoTime();
            kernel.run(batch);
            long t1 = System.nanoTime();
            System.out.println("measured," + r + "," + (t1 - t0) + "," + batch + "," + kernel.checksum() + "," + kernel.nnzOut());
        }

        // ---- 5. conversion cost COO -> CSR (sparse only, measured separately)
        if (cooA != null) {
            for (int r = 0; r < reps; r++) {
                long t0 = System.nanoTime();
                Csr a = Csr.fromCoo(n, cooA.r(), cooA.c(), cooA.v());
                Csr b = Csr.fromCoo(n, cooB.r(), cooB.c(), cooB.v());
                long t1 = System.nanoTime();
                System.out.println("convert," + r + "," + (t1 - t0) + ",1," + (a.nnz() + b.nnz()) + ",0");
            }
            storage += ((SparseKernel) kernel).C.storageBytes();
        }

        // ---- output and metadata
        if (out != null) {
            double[] C;
            if (kernel instanceof SparseKernel sk) {
                C = new double[n * n];
                sk.C.toDense(C);
            } else {
                C = ((DenseKernel) kernel).C;
            }
            saveDense(out, C);
        }
        Runtime rt = Runtime.getRuntime();
        System.out.println("#array_storage_bytes," + storage);
        System.out.println("#peak_rss_bytes," + peakRssBytes());
        System.out.println("#java_heap_used_bytes," + (rt.totalMemory() - rt.freeMemory()));
    }
}
