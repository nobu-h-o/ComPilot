# Best schedules: code before and after

Best schedule found for each kernel in the `medium5_vanilla_epyc` run (`claude-haiku-5-5`, PolyBench MEDIUM, 2026-11-23).

| Kernel | Before (ms) | After (ms) | Speedup | Transformations |
|---|---|---|---|---|
| [2mm](#2mm) | 11.7148 | 0.5931 | 19.75x | tile, parallelize, unroll |
| [syr2k](#syr2k) | 4.7798 | 0.3356 | 14.24x | interchange, parallelize |
| [heat3d](#heat3d) | 11.3549 | 6.6130 | 1.72x | tile, parallelize |
| [symm](#symm) | 8.7799 | 7.5828 | 1.16x | unroll |
| [trisolv](#trisolv) | 0.0696 | 0.0693 | 1.01x | unroll |

The **C code** is a readable translation of the compiler's lowered IR, with array names from the benchmark sources. The kernels are Tiramisu programs, so the compiler never produces this C directly. `#pragma omp parallel for` stands for the IR's `parallel` loops, and `min`/`max` are the usual functions.

The **lowered IR** is copied from `results/medium5_vanilla_epyc.log`, with assertions, buffer setup and allocations removed. It is what was executed. Loops are written `for (var, start, count)`, so the third value is an iteration count, not an upper bound. `unrolled` loops are expanded by the compiler.

Levels are counted from the outermost loop (`L0`); `L-1` is the innermost.

## 2mm

**Schedule:** `comp01.Tile2D(L0,L1,4,4)+comp00.Parallelize(L0)+comp01.Parallelize(L0)+comp02.Parallelize(L0)+comp03.Parallelize(L0)+comp01.Unroll(L-1,10)+comp03.Unroll(L-1,5)`

**Result:** 11.7148 ms to 0.5931 ms (19.75x)

### Before

```c
for (int i = 0; i < 180; i++)
  for (int j = 0; j < 190; j++) {
    tmp[i][j] = 0.0;                                  // comp00
    for (int k = 0; k < 210; k++)
      tmp[i][j] += 1.5 * A[i][k] * B[k][j];           // comp01
  }

for (int i = 0; i < 180; i++)
  for (int j = 0; j < 220; j++) {
    D[i][j] *= 1.2;                                   // comp02
    for (int k = 0; k < 190; k++)
      D[i][j] += tmp[i][k] * C[k][j];                 // comp03
  }
```

### After

```c
#pragma omp parallel for
for (int it = 0; it < 45; it++)                       // i tiles of 4
  for (int jt = 0; jt < 48; jt++)                     // j tiles of 4 (last has 2)
    for (int i = 4*it; i < 4*it + 4; i++)
      for (int j = 4*jt; j < min(4*jt + 4, 190); j++) {
        tmp[i][j] = 0.0;                              // comp00
        for (int k = 0; k < 210; k += 10) {           // unrolled x10
          tmp[i][j] += 1.5 * A[i][k+0] * B[k+0][j];
          tmp[i][j] += 1.5 * A[i][k+1] * B[k+1][j];
          tmp[i][j] += 1.5 * A[i][k+2] * B[k+2][j];
          tmp[i][j] += 1.5 * A[i][k+3] * B[k+3][j];
          tmp[i][j] += 1.5 * A[i][k+4] * B[k+4][j];
          tmp[i][j] += 1.5 * A[i][k+5] * B[k+5][j];
          tmp[i][j] += 1.5 * A[i][k+6] * B[k+6][j];
          tmp[i][j] += 1.5 * A[i][k+7] * B[k+7][j];
          tmp[i][j] += 1.5 * A[i][k+8] * B[k+8][j];
          tmp[i][j] += 1.5 * A[i][k+9] * B[k+9][j];
        }
      }

#pragma omp parallel for
for (int i = 0; i < 180; i++)
  for (int j = 0; j < 220; j++) {
    D[i][j] *= 1.2;                                   // comp02
    for (int k = 0; k < 190; k += 5) {                // unrolled x5
      D[i][j] += tmp[i][k+0] * C[k+0][j];
      D[i][j] += tmp[i][k+1] * C[k+1][j];
      D[i][j] += tmp[i][k+2] * C[k+2][j];
      D[i][j] += tmp[i][k+3] * C[k+3][j];
      D[i][j] += tmp[i][k+4] * C[k+4][j];
    }
  }
```

### What changed

- `comp01.Tile2D(L0,L1,4,4)` tiles the `i`/`j` loops. `comp00` shares them, so it is tiled too.
- The `Parallelize(L0)` commands make the tile loop `it` (first nest) and `i` (second nest) parallel. The commands for comp00/comp01 target the same loop, as do those for comp02/comp03.
- `L-1` is the innermost `k` loop. Its trip counts (210 and 190) divide evenly by 10 and 5, so there is no remainder loop.
- The second nest is not tiled.

<details>
<summary>Lowered IR</summary>

Before:

```text
produce  {
 for (c1, 0, 180) {
  for (c3, 0, 190) {
   b_tmp[(c1*190) + c3] = 0.000000
   for (c5, 0, 210) {
    b_tmp[(c1*190) + c3] = b_tmp[(c1*190) + c3] + ((b_A[(c1*210) + c5]*b_B[(c5*190) + c3])*1.500000)
   }
  }
 }
 for (c1, 0, 180) {
  for (c3, 0, 220) {
   b_D[(c1*220) + c3] = b_D[(c1*220) + c3]*1.200000
   for (c5, 0, 190) {
    b_D[(c1*220) + c3] = b_D[(c1*220) + c3] + (b_tmp[(c1*190) + c5]*b_C[(c5*220) + c3])
   }
  }
 }
}
```

After (reported at log line 1255):

```text
produce  {
 parallel (c1, 0, 45) {
  for (c3, 0, 48) {
   for (c5, 0, 4) {
    for (c7, 0, min(c3*-4, -186) + 190) {
     b_tmp[(((c1*4) + c5)*190) + ((c3*4) + c7)] = 0.000000
     for (c9, 0, 21) {
      unrolled (c11, 0, 10) {
       b_tmp[(((c1*4) + c5)*190) + ((c3*4) + c7)] = b_tmp[(((c1*4) + c5)*190) + ((c3*4) + c7)] + ((b_A[(((c1*4) + c5)*210) + ((c9*10) + c11)]*b_B[(((c9*10) + c11)*190) + ((c3*4) + c7)])*1.500000)
      }
     }
    }
   }
  }
 }
 parallel (c1, 0, 180) {
  for (c3, 0, 220) {
   b_D[(c1*220) + c3] = b_D[(c1*220) + c3]*1.200000
   for (c5, 0, 38) {
    unrolled (c7, 0, 5) {
     b_D[(c1*220) + c3] = b_D[(c1*220) + c3] + (b_tmp[(c1*190) + ((c5*5) + c7)]*b_C[(((c5*5) + c7)*220) + c3])
    }
   }
  }
 }
}
```

</details>

## syr2k

**Schedule:** `comp01.Interchange(L1,L2)+comp01.Parallelize(L0)`

**Result:** 4.7798 ms to 0.3356 ms (14.24x)

### Before

```c
for (int i = 0; i < 240; i++) {
  for (int j = 0; j <= i; j++)
    C[i][j] *= 1.2;                                   // comp00
  for (int k = 0; k < 200; k++)
    for (int j = 0; j <= i; j++)
      C[i][j] += 1.5*A[j][k]*B[i][k] + 1.5*B[j][k]*A[i][k];   // comp01
}
```

### After

```c
#pragma omp parallel for
for (int i = 0; i < 240; i++) {
  for (int j = 0; j <= i; j++)
    C[i][j] *= 1.2;                                   // comp00
  for (int j = 0; j <= i; j++)                        // j and k swapped
    for (int k = 0; k < 200; k++)
      C[i][j] += 1.5*A[j][k]*B[i][k] + 1.5*B[j][k]*A[i][k];   // comp01
}
```

### What changed

- `Interchange(L1,L2)` makes `k` innermost, so all four `A`/`B` accesses become stride-1 and `C[i][j]` is fixed in the inner loop. Before, `A[j][k]` and `B[j][k]` jumped a whole row per iteration.
- `Parallelize(L0)` makes `i` parallel. comp00 is in the same `i` loop, so it runs in parallel too.
- The `i` loop is triangular, so work per iteration grows with `i`.

<details>
<summary>Lowered IR</summary>

Before:

```text
produce  {
 for (c1, 0, 240) {
  for (c3, 0, c1 + 1) {
   b_C[(c1*240) + c3] = b_C[(c1*240) + c3]*1.200000
  }
  for (c3, 0, 200) {
   for (c5, 0, c1 + 1) {
    b_C[(c1*240) + c5] = ((b_B[(c5*200) + c3]*b_A[(c1*200) + c3])*1.500000) + (b_C[(c1*240) + c5] + ((b_A[(c5*200) + c3]*b_B[(c1*200) + c3])*1.500000))
   }
  }
 }
}
```

After (reported at log line 4709):

```text
produce  {
 parallel (c1, 0, 240) {
  for (c3, 0, c1 + 1) {
   b_C[(c1*240) + c3] = b_C[(c1*240) + c3]*1.200000
  }
  for (c3, 0, c1 + 1) {
   for (c5, 0, 200) {
    b_C[(c1*240) + c3] = ((b_B[(c3*200) + c5]*b_A[(c1*200) + c5])*1.500000) + (b_C[(c1*240) + c3] + ((b_A[(c3*200) + c5]*b_B[(c1*200) + c5])*1.500000))
   }
  }
 }
}
```

</details>

## heat3d

**Schedule:** `comp00.Tile2D(L1,L2,19,19)+comp01.Tile2D(L1,L2,19,19)+comp00.Parallelize(L1)+comp01.Parallelize(L1)`

**Result:** 11.3549 ms to 6.6130 ms (1.72x)

### Before

```c
for (int t = 0; t < 100; t++) {
  for (int i = 1; i < 39; i++)
    for (int j = 1; j < 39; j++)
      for (int k = 1; k < 39; k++)                    // comp00
        B[i][j][k] = A[i][j][k] + 0.125 * (
            (A[i+1][j][k] - 2.0*A[i][j][k] + A[i-1][j][k])
          + (A[i][j+1][k] - 2.0*A[i][j][k] + A[i][j-1][k])
          + (A[i][j][k+1] - 2.0*A[i][j][k] + A[i][j][k-1]));

  for (int i = 1; i < 39; i++)
    for (int j = 1; j < 39; j++)
      for (int k = 1; k < 39; k++)                    // comp01
        A[i][j][k] = B[i][j][k] + 0.125 * (
            (B[i+1][j][k] - 2.0*B[i][j][k] + B[i-1][j][k])
          + (B[i][j+1][k] - 2.0*B[i][j][k] + B[i][j-1][k])
          + (B[i][j][k+1] - 2.0*B[i][j][k] + B[i][j][k-1]));
}
```

### After

```c
for (int t = 0; t < 100; t++) {
  #pragma omp parallel for
  for (int it = 0; it < 3; it++)                      // i tiles of 19
    for (int jt = 0; jt < 3; jt++)                    // j tiles of 19
      for (int i = max(19*it, 1); i < min(19*it + 19, 39); i++)
        for (int j = max(19*jt, 1); j < min(19*jt + 19, 39); j++)
          for (int k = 1; k < 39; k++)                // comp00
        B[i][j][k] = A[i][j][k] + 0.125 * (
            (A[i+1][j][k] - 2.0*A[i][j][k] + A[i-1][j][k])
          + (A[i][j+1][k] - 2.0*A[i][j][k] + A[i][j-1][k])
          + (A[i][j][k+1] - 2.0*A[i][j][k] + A[i][j][k-1]));

  #pragma omp parallel for
  for (int it = 0; it < 3; it++)                      // i tiles of 19
    for (int jt = 0; jt < 3; jt++)                    // j tiles of 19
      for (int i = max(19*it, 1); i < min(19*it + 19, 39); i++)
        for (int j = max(19*jt, 1); j < min(19*jt + 19, 39); j++)
          for (int k = 1; k < 39; k++)                // comp01
        A[i][j][k] = B[i][j][k] + 0.125 * (
            (B[i+1][j][k] - 2.0*B[i][j][k] + B[i-1][j][k])
          + (B[i][j+1][k] - 2.0*B[i][j][k] + B[i][j-1][k])
          + (B[i][j][k+1] - 2.0*B[i][j][k] + B[i][j][k-1]));
}
```

### What changed

- `Tile2D(L1,L2,19,19)` tiles the `i`/`j` loops of both stencils.
- `Parallelize(L1)` makes the `i` tile loop parallel, inside the sequential time loop.
- Tiles start at multiples of 19, but `i` and `j` run from 1 to 38. The three `i` tiles therefore cover 18, 19 and 1 values (`1-18`, `19-37`, `38`). Two tiles hold nearly all the work, which caps the ideal parallel speedup at 2x (38/19).
- Parallelizing the untiled `i` loops (`comp00.Parallelize(L1)+comp01.Parallelize(L1)`) took 46.74 ms, slower than the 11.35 ms baseline.

<details>
<summary>Lowered IR</summary>

Before:

```text
produce  {
 for (c1, 0, 100) {
  for (c3, 1, 38) {
   for (c5, 1, 38) {
    for (c7, 1, 38) {
     b_B[(c3*1600) + ((c5*40) + c7)] = b_A[(c3*1600) + ((c5*40) + c7)] + ((((b_A[((c3*1600) + ((c5*40) + c7)) + -1600] + (b_A[((c3*1600) + ((c5*40) + c7)) + 1600] - (b_A[(c3*1600) + ((c5*40) + c7)]*2.000000))) + (b_A[((c3*1600) + ((c5*40) + c7)) + -40] + (b_A[((c3*1600) + ((c5*40) + c7)) + 40] - (b_A[(c3*1600) + ((c5*40) + c7)]*2.000000)))) + (b_A[((c3*1600) + ((c5*40) + c7)) + -1] + (b_A[((c3*1600) + ((c5*40) + c7)) + 1] - (b_A[(c3*1600) + ((c5*40) + c7)]*2.000000))))*0.125000)
    }
   }
  }
  for (c3, 1, 38) {
   for (c5, 1, 38) {
    for (c7, 1, 38) {
     b_A[(c3*1600) + ((c5*40) + c7)] = b_B[(c3*1600) + ((c5*40) + c7)] + ((((b_B[((c3*1600) + ((c5*40) + c7)) + -1600] + (b_B[((c3*1600) + ((c5*40) + c7)) + 1600] - (b_B[(c3*1600) + ((c5*40) + c7)]*2.000000))) + (b_B[((c3*1600) + ((c5*40) + c7)) + -40] + (b_B[((c3*1600) + ((c5*40) + c7)) + 40] - (b_B[(c3*1600) + ((c5*40) + c7)]*2.000000)))) + (b_B[((c3*1600) + ((c5*40) + c7)) + -1] + (b_B[((c3*1600) + ((c5*40) + c7)) + 1] - (b_B[(c3*1600) + ((c5*40) + c7)]*2.000000))))*0.125000)
    }
   }
  }
 }
}
```

After (reported at log line 2815):

```text
produce  {
 for (c1, 0, 100) {
  parallel (c3, 0, 3) {
   for (c5, 0, 3) {
    for (c7, max(c3*-19, -1) + 1, min(min(c3*19, (c3*-19) + 21), 1) + 18) {
     for (c9, max(c5*-19, -1) + 1, min(min(c5*19, (c5*-19) + 21), 1) + 18) {
      for (c11, 1, 38) {
       b_B[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)] = b_A[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)] + ((((b_A[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + -1600] + (b_A[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + 1600] - (b_A[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)]*2.000000))) + (b_A[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + -40] + (b_A[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + 40] - (b_A[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)]*2.000000)))) + (b_A[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + -1] + (b_A[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + 1] - (b_A[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)]*2.000000))))*0.125000)
      }
     }
    }
   }
  }
  parallel (c3, 0, 3) {
   for (c5, 0, 3) {
    for (c7, max(c3*-19, -1) + 1, min(min(c3*19, (c3*-19) + 21), 1) + 18) {
     for (c9, max(c5*-19, -1) + 1, min(min(c5*19, (c5*-19) + 21), 1) + 18) {
      for (c11, 1, 38) {
       b_A[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)] = b_B[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)] + ((((b_B[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + -1600] + (b_B[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + 1600] - (b_B[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)]*2.000000))) + (b_B[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + -40] + (b_B[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + 40] - (b_B[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)]*2.000000)))) + (b_B[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + -1] + (b_B[((((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)) + 1] - (b_B[(((c3*19) + c7)*1600) + ((((c5*19) + c9)*40) + c11)]*2.000000))))*0.125000)
      }
     }
    }
   }
  }
 }
}
```

</details>

## symm

**Schedule:** `comp01.Unroll(L2,4)`

**Result:** 8.7799 ms to 7.5828 ms (1.16x)

### Before

```c
for (int i = 0; i < 200; i++)
  for (int j = 0; j < 240; j++) {
    temp = 0.0;                                       // comp00
    for (int k = 0; k < i; k++) {                     // comp01 (both statements)
      C[k][j] += 1.5 * A[i][k] * B[i][j];
      temp    += A[i][k] * B[k][j];
    }
    C[i][j] = 1.2*C[i][j] + 1.5*B[i][j]*A[i][i] + 1.5*temp;
  }
```

### After

```c
for (int i = 0; i < 200; i++)
  for (int j = 0; j < 240; j++) {
    temp = 0.0;                                       // comp00
    int k = 0;
    for (; k < i/4*4; k += 4) {                       // unrolled x4
      C[k+0][j] += 1.5 * A[i][k+0] * B[i][j];
      temp += A[i][k+0] * B[k+0][j];
      C[k+1][j] += 1.5 * A[i][k+1] * B[i][j];
      temp += A[i][k+1] * B[k+1][j];
      C[k+2][j] += 1.5 * A[i][k+2] * B[i][j];
      temp += A[i][k+2] * B[k+2][j];
      C[k+3][j] += 1.5 * A[i][k+3] * B[i][j];
      temp += A[i][k+3] * B[k+3][j];
    }
    for (; k < i; k++) {                              // remainder: i % 4 iterations
      C[k][j] += 1.5 * A[i][k] * B[i][j];
      temp    += A[i][k] * B[k][j];
    }
    C[i][j] = 1.2*C[i][j] + 1.5*B[i][j]*A[i][i] + 1.5*temp;
  }
```

### What changed

- `comp01` is a block of two statements, and `Unroll(L2,4)` unrolls both.
- The trip count `i` varies, so the compiler adds a remainder loop.
- No parallelism, tiling or interchange is applied.

<details>
<summary>Lowered IR</summary>

Before:

```text
produce  {
 for (c1, 0, 200) {
  for (c3, 0, 240) {
   b_temp[0] = 0.000000
   for (c5, 0, c1) {
    b_C[(c5*240) + c3] = b_C[(c5*240) + c3] + ((b_B[(c1*240) + c3]*b_A[(c1*200) + c5])*1.500000)
    b_temp[0] = b_temp[0] + (b_A[(c1*200) + c5]*b_B[(c5*240) + c3])
   }
   b_C[(c1*240) + c3] = (b_temp[0]*1.500000) + ((b_C[(c1*240) + c3]*1.200000) + ((b_B[(c1*240) + c3]*b_A[c1*201])*1.500000))
  }
 }
}
```

After (reported at log line 3261):

```text
produce  {
 for (c1, 0, 200) {
  for (c3, 0, 240) {
   b_temp[0] = 0.000000
   for (c5, 0, int32((float32)floor_f32(float32((c1/4))))) {
    unrolled (c7, 0, 4) {
     b_C[(((c5*4) + c7)*240) + c3] = b_C[(((c5*4) + c7)*240) + c3] + ((b_B[(c1*240) + c3]*b_A[(c1*200) + ((c5*4) + c7)])*1.500000)
     b_temp[0] = b_temp[0] + (b_A[(c1*200) + ((c5*4) + c7)]*b_B[(((c5*4) + c7)*240) + c3])
    }
   }
   for (c5, (c1/4)*4, c1 % 4) {
    b_C[(c5*240) + c3] = b_C[(c5*240) + c3] + ((b_B[(c1*240) + c3]*b_A[(c1*200) + c5])*1.500000)
    b_temp[0] = b_temp[0] + (b_A[(c1*200) + c5]*b_B[(c5*240) + c3])
   }
   b_C[(c1*240) + c3] = (b_temp[0]*1.500000) + ((b_C[(c1*240) + c3]*1.200000) + ((b_B[(c1*240) + c3]*b_A[c1*201])*1.500000))
  }
 }
}
```

</details>

## trisolv

**Schedule:** `comp01.Unroll(L1,4)`

**Result:** 0.0696 ms to 0.0693 ms (1.01x)

### Before

```c
for (int i = 0; i < 400; i++) {
  x[i] = b[i];                                        // comp00
  for (int j = 0; j < i; j++)
    x[i] -= L[i][j] * x[j];                           // comp01
  x[i] = x[i] / L[i][i];
}
```

### After

```c
for (int i = 0; i < 400; i++) {
  x[i] = b[i];                                        // comp00
  int j = 0;
  for (; j < i/4*4; j += 4) {                         // unrolled x4
    x[i] -= L[i][j+0] * x[j+0];
    x[i] -= L[i][j+1] * x[j+1];
    x[i] -= L[i][j+2] * x[j+2];
    x[i] -= L[i][j+3] * x[j+3];
  }
  for (; j < i; j++)                                  // remainder: i % 4 iterations
    x[i] -= L[i][j] * x[j];
  x[i] = x[i] / L[i][i];
}
```

### What changed

- Only the inner loop is unrolled, with a remainder loop for the varying trip count.
- The 0.5% difference on a 0.07 ms kernel is probably timing noise.

<details>
<summary>Lowered IR</summary>

Before:

```text
produce  {
 for (c1, 0, 400) {
  b_x[c1] = b_b[c1]
  for (c3, 0, c1) {
   b_x[c1] = b_x[c1] - (b_L[(c1*400) + c3]*b_x[c3])
  }
  b_x[c1] = b_x[c1]/b_L[c1*401]
 }
}
```

After (reported at log line 5821):

```text
produce  {
 for (c1, 0, 400) {
  b_x[c1] = b_b[c1]
  for (c3, 0, int32((float32)floor_f32(float32((c1/4))))) {
   unrolled (c5, 0, 4) {
    b_x[c1] = b_x[c1] - (b_L[(c1*400) + ((c3*4) + c5)]*b_x[(c3*4) + c5])
   }
  }
  for (c3, (c1/4)*4, c1 % 4) {
   b_x[c1] = b_x[c1] - (b_L[(c1*400) + c3]*b_x[c3])
  }
  b_x[c1] = b_x[c1]/b_L[c1*401]
 }
}
```

</details>
