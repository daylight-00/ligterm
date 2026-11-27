# Environment of the stored results

| | |
|---|---|
| Python | 3.12.3 |
| numpy / scipy / scikit-learn | 2.5.2 / 1.18.1 / 1.9.1 |
| torch | 2.14.0, CPU build (scripts 20 to 22 and 42) |
| Operating system | Linux (Ubuntu), x86-64; Intel Xeon CPUs |
| Threads | 2 CPUs and 2 BLAS threads (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2`) for the per-study scripts 10 to 40; 8 threads for the refit of LigTerm-GB (41) and for the refits of the historical test (44); 4 processes with one BLAS thread each for 50 and 51 |

The number of BLAS threads is part of the numerical result of the logistic-regression fits: set it as above when a result must equal the expected table exactly. With other versions or threads, expect differences in the last digits; the retrained NetCleave-style networks (torch) agree to about 3e-06 across machines and are accepted within 1e-4 by `scripts/90_check_reproduction.py`.

Measured peak memory of the runs: at most 3.2 GB per task for scripts 10 to 41, 50 and 51, and 2.5 to 4.9 GB for the refits of the historical test (44). The table in [README.md](README.md) gives the run times.
