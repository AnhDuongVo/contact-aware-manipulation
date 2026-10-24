# Generated result tables

Produced by `cambench/analysis/report.py`. Intervals are 95% hierarchical bootstrap intervals (seeds, then shared episodes).

## insertion: safe success

| condition | in-distribution | held-out (mean) |
|---|---|---|
| vision/state | 0.82 [0.75, 0.89] | 0.59 [0.53, 0.64] |
| + proprio | 0.99 [0.96, 1.00] | 0.91 [0.88, 0.94] |
| + wrist F/T | 0.98 [0.95, 1.00] | 0.90 [0.86, 0.93] |
| + proprio + F/T | 0.95 [0.87, 1.00] | 0.87 [0.82, 0.91] |
| + proprio + contact aux | 0.99 [0.97, 1.00] | 0.90 [0.87, 0.94] |
| + proprio + F/T + contact aux | 0.97 [0.93, 0.99] | 0.87 [0.82, 0.90] |
| + proprio + F/T + privileged aux | 0.97 [0.94, 1.00] | 0.88 [0.85, 0.91] |
| scripted expert | 1.00 [1.00, 1.00] | 0.92 [0.89, 0.95] |

## press: safe success

| condition | in-distribution | held-out (mean) |
|---|---|---|
| vision/state | 0.93 [0.88, 0.98] | 0.82 [0.78, 0.87] |
| + proprio | 0.81 [0.73, 0.89] | 0.63 [0.57, 0.70] |
| + wrist F/T | 0.75 [0.66, 0.83] | 0.60 [0.54, 0.66] |
| + proprio + F/T | 0.78 [0.69, 0.86] | 0.61 [0.55, 0.68] |
| + proprio + contact aux | 0.67 [0.57, 0.78] | 0.50 [0.40, 0.62] |
| + proprio + F/T + contact aux | 0.65 [0.48, 0.82] | 0.50 [0.38, 0.64] |
| + proprio + F/T + privileged aux | 0.89 [0.80, 0.97] | 0.74 [0.63, 0.83] |
| scripted expert | 0.98 [0.95, 1.00] | 1.00 [1.00, 1.00] |

## surface: safe success

| condition | in-distribution | held-out (mean) |
|---|---|---|
| vision/state | 0.06 [0.03, 0.10] | 0.04 [0.01, 0.07] |
| + proprio | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| + wrist F/T | 0.99 [0.97, 1.00] | 0.98 [0.96, 0.99] |
| + proprio + F/T | 0.99 [0.97, 1.00] | 0.99 [0.97, 1.00] |
| + proprio + contact aux | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] |
| + proprio + F/T + contact aux | 1.00 [0.98, 1.00] | 0.98 [0.96, 1.00] |
| + proprio + F/T + privileged aux | 1.00 [0.98, 1.00] | 0.99 [0.97, 1.00] |
| scripted expert | 1.00 [1.00, 1.00] | 0.96 [0.94, 0.99] |

## Force and timing (ID + held-out pooled)

| task | condition | peak force [N] | overshoot [N] | time to success [s] |
|---|---|---|---|---|
| insertion | vision/state | 22.14 [20.38, 23.96] | -0.26 [-0.97, 0.44] | 2.67 [2.58, 2.76] |
| insertion | + proprio | 6.37 [6.19, 6.55] | 4.40 [4.12, 4.67] | 2.70 [2.60, 2.81] |
| insertion | + wrist F/T | 6.22 [6.04, 6.39] | 4.83 [4.56, 5.08] | 2.47 [2.41, 2.54] |
| insertion | + proprio + F/T | 6.20 [5.93, 6.44] | 4.76 [4.50, 5.04] | 2.56 [2.46, 2.67] |
| insertion | + proprio + contact aux | 6.12 [5.94, 6.28] | 3.76 [3.49, 4.02] | 2.82 [2.72, 2.93] |
| insertion | + proprio + F/T + contact aux | 6.28 [6.14, 6.43] | 4.21 [3.91, 4.51] | 2.63 [2.55, 2.72] |
| insertion | + proprio + F/T + privileged aux | 6.55 [6.35, 6.74] | 4.56 [4.30, 4.83] | 2.67 [2.58, 2.77] |
| insertion | scripted expert | 6.12 [5.95, 6.29] | 4.80 [4.55, 5.05] | 2.41 [2.35, 2.47] |
| press | vision/state | 14.58 [13.43, 15.85] | -2.82 [-2.92, -2.71] | 4.04 [3.91, 4.17] |
| press | + proprio | 10.51 [10.15, 10.92] | -3.08 [-3.21, -2.95] | 4.01 [3.83, 4.20] |
| press | + wrist F/T | 11.24 [10.73, 11.82] | -3.36 [-3.51, -3.22] | 3.96 [3.83, 4.10] |
| press | + proprio + F/T | 11.12 [10.81, 11.44] | -3.29 [-3.44, -3.17] | 3.89 [3.74, 4.02] |
| press | + proprio + contact aux | 11.53 [10.63, 12.67] | -2.42 [-2.48, -2.36] | 4.16 [3.95, 4.39] |
| press | + proprio + F/T + contact aux | 10.97 [10.21, 11.61] | -2.56 [-2.63, -2.49] | 4.11 [3.96, 4.28] |
| press | + proprio + F/T + privileged aux | 11.04 [10.23, 11.95] | -3.34 [-3.46, -3.21] | 3.60 [3.50, 3.73] |
| press | scripted expert | 10.25 [9.83, 10.66] | -3.53 [-3.66, -3.40] | 3.48 [3.44, 3.53] |
| surface | vision/state | 22.50 [18.73, 25.98] | -1.76 [-2.39, -1.17] | 9.49 [9.43, 9.54] |
| surface | + proprio | 13.33 [13.03, 13.64] | 0.74 [0.57, 0.92] | 9.57 [9.52, 9.61] |
| surface | + wrist F/T | 15.79 [15.57, 16.01] | 1.05 [0.92, 1.17] | 9.63 [9.58, 9.67] |
| surface | + proprio + F/T | 15.73 [15.52, 15.95] | 0.95 [0.81, 1.09] | 9.61 [9.56, 9.67] |
| surface | + proprio + contact aux | 12.79 [12.61, 12.97] | 0.41 [0.20, 0.64] | 9.54 [9.48, 9.59] |
| surface | + proprio + F/T + contact aux | 15.40 [15.13, 15.67] | 0.96 [0.82, 1.12] | 9.53 [9.45, 9.61] |
| surface | + proprio + F/T + privileged aux | 14.94 [14.74, 15.15] | 1.06 [0.81, 1.35] | 9.51 [9.45, 9.58] |
| surface | scripted expert | 16.00 [15.76, 16.26] | 0.83 [0.70, 0.96] | 9.62 [9.61, 9.64] |

## Paired differences (a - b)

| task | eval | comparison | metric | diff | 95% CI |
|---|---|---|---|---|---|
| insertion | id | proprioception over vision/state | safe_success | +0.167 * | [+0.093, +0.247] |
| insertion | id | proprioception over vision/state | peak_force | -9.395 * | [-12.541, -6.610] |
| insertion | id | wrist F/T over vision/state | safe_success | +0.160 * | [+0.097, +0.227] |
| insertion | id | wrist F/T over vision/state | peak_force | -9.479 * | [-12.584, -6.758] |
| insertion | id | wrist F/T on top of proprioception | safe_success | -0.040 | [-0.110, +0.000] |
| insertion | id | wrist F/T on top of proprioception | peak_force | -0.217 | [-0.549, +0.110] |
| insertion | id | contact-prediction aux without F/T input | safe_success | +0.000 | [-0.020, +0.027] |
| insertion | id | contact-prediction aux without F/T input | peak_force | -0.223 | [-0.537, +0.093] |
| insertion | id | contact-prediction aux with F/T input | safe_success | +0.017 | [-0.040, +0.100] |
| insertion | id | contact-prediction aux with F/T input | peak_force | +0.253 | [-0.198, +0.710] |
| insertion | id | F/T as input vs. F/T only as training signal | safe_success | -0.040 | [-0.117, +0.010] |
| insertion | id | F/T as input vs. F/T only as training signal | peak_force | +0.006 | [-0.425, +0.408] |
| insertion | id | privileged vs. self-supervised aux labels | safe_success | +0.007 | [-0.020, +0.037] |
| insertion | id | privileged vs. self-supervised aux labels | peak_force | +0.257 | [-0.109, +0.609] |
| insertion | heldout | proprioception over vision/state | safe_success | +0.323 * | [+0.267, +0.377] |
| insertion | heldout | proprioception over vision/state | peak_force | -17.885 * | [-19.835, -15.865] |
| insertion | heldout | wrist F/T over vision/state | safe_success | +0.312 * | [+0.257, +0.367] |
| insertion | heldout | wrist F/T over vision/state | peak_force | -18.064 * | [-20.006, -16.028] |
| insertion | heldout | wrist F/T on top of proprioception | safe_success | -0.043 * | [-0.081, -0.011] |
| insertion | heldout | wrist F/T on top of proprioception | peak_force | -0.161 | [-0.366, +0.035] |
| insertion | heldout | contact-prediction aux without F/T input | safe_success | -0.008 | [-0.043, +0.027] |
| insertion | heldout | contact-prediction aux without F/T input | peak_force | -0.267 * | [-0.452, -0.082] |
| insertion | heldout | contact-prediction aux with F/T input | safe_success | -0.001 | [-0.037, +0.034] |
| insertion | heldout | contact-prediction aux with F/T input | peak_force | +0.027 | [-0.190, +0.242] |
| insertion | heldout | F/T as input vs. F/T only as training signal | safe_success | -0.036 | [-0.083, +0.009] |
| insertion | heldout | F/T as input vs. F/T only as training signal | peak_force | +0.106 | [-0.177, +0.372] |
| insertion | heldout | privileged vs. self-supervised aux labels | safe_success | +0.013 | [-0.027, +0.051] |
| insertion | heldout | privileged vs. self-supervised aux labels | peak_force | +0.270 * | [+0.025, +0.526] |
| press | id | proprioception over vision/state | safe_success | -0.120 * | [-0.193, -0.050] |
| press | id | proprioception over vision/state | peak_force | -1.143 | [-2.977, +0.462] |
| press | id | wrist F/T over vision/state | safe_success | -0.187 * | [-0.270, -0.103] |
| press | id | wrist F/T over vision/state | peak_force | -0.529 | [-2.392, +1.176] |
| press | id | wrist F/T on top of proprioception | safe_success | -0.037 | [-0.097, +0.027] |
| press | id | wrist F/T on top of proprioception | peak_force | +0.476 * | [+0.017, +0.935] |
| press | id | contact-prediction aux without F/T input | safe_success | -0.140 * | [-0.240, -0.033] |
| press | id | contact-prediction aux without F/T input | peak_force | +0.941 * | [+0.010, +2.310] |
| press | id | contact-prediction aux with F/T input | safe_success | -0.127 | [-0.303, +0.060] |
| press | id | contact-prediction aux with F/T input | peak_force | -0.196 | [-0.639, +0.340] |
| press | id | F/T as input vs. F/T only as training signal | safe_success | +0.103 | [-0.020, +0.217] |
| press | id | F/T as input vs. F/T only as training signal | peak_force | -0.464 | [-1.844, +0.448] |
| press | id | privileged vs. self-supervised aux labels | safe_success | +0.243 * | [+0.133, +0.357] |
| press | id | privileged vs. self-supervised aux labels | peak_force | -0.207 | [-1.090, +0.654] |
| press | heldout | proprioception over vision/state | safe_success | -0.193 * | [-0.280, -0.111] |
| press | heldout | proprioception over vision/state | peak_force | -5.245 * | [-6.808, -3.779] |
| press | heldout | wrist F/T over vision/state | safe_success | -0.227 * | [-0.307, -0.149] |
| press | heldout | wrist F/T over vision/state | peak_force | -4.467 * | [-6.024, -2.996] |
| press | heldout | wrist F/T on top of proprioception | safe_success | -0.019 | [-0.071, +0.033] |
| press | heldout | wrist F/T on top of proprioception | peak_force | +0.669 * | [+0.290, +1.139] |
| press | heldout | contact-prediction aux without F/T input | safe_success | -0.131 * | [-0.207, -0.052] |
| press | heldout | contact-prediction aux without F/T input | peak_force | +1.064 * | [+0.122, +2.450] |
| press | heldout | contact-prediction aux with F/T input | safe_success | -0.111 | [-0.216, +0.032] |
| press | heldout | contact-prediction aux with F/T input | peak_force | -0.132 | [-0.723, +0.533] |
| press | heldout | F/T as input vs. F/T only as training signal | safe_success | +0.112 | [-0.004, +0.208] |
| press | heldout | F/T as input vs. F/T only as training signal | peak_force | -0.394 | [-1.885, +0.773] |
| press | heldout | privileged vs. self-supervised aux labels | safe_success | +0.236 * | [+0.176, +0.297] |
| press | heldout | privileged vs. self-supervised aux labels | peak_force | +0.175 | [-1.177, +1.183] |
| surface | id | proprioception over vision/state | safe_success | +0.940 * | [+0.897, +0.977] |
| surface | id | proprioception over vision/state | peak_force | -8.069 * | [-12.180, -4.119] |
| surface | id | wrist F/T over vision/state | safe_success | +0.930 * | [+0.883, +0.970] |
| surface | id | wrist F/T over vision/state | peak_force | -5.390 * | [-9.497, -1.395] |
| surface | id | wrist F/T on top of proprioception | safe_success | -0.007 | [-0.027, +0.000] |
| surface | id | wrist F/T on top of proprioception | peak_force | +2.752 * | [+2.376, +3.140] |
| surface | id | contact-prediction aux without F/T input | safe_success | +0.000 | [+0.000, +0.000] |
| surface | id | contact-prediction aux without F/T input | peak_force | -0.461 * | [-0.805, -0.200] |
| surface | id | contact-prediction aux with F/T input | safe_success | +0.003 | [-0.013, +0.027] |
| surface | id | contact-prediction aux with F/T input | peak_force | -0.502 * | [-0.841, -0.199] |
| surface | id | F/T as input vs. F/T only as training signal | safe_success | -0.007 | [-0.027, +0.000] |
| surface | id | F/T as input vs. F/T only as training signal | peak_force | +3.213 * | [+2.957, +3.479] |
| surface | id | privileged vs. self-supervised aux labels | safe_success | +0.000 | [-0.013, +0.013] |
| surface | id | privileged vs. self-supervised aux labels | peak_force | -0.363 * | [-0.634, -0.117] |
| surface | heldout | proprioception over vision/state | safe_success | +0.963 * | [+0.933, +0.985] |
| surface | heldout | proprioception over vision/state | peak_force | -9.729 * | [-13.376, -6.077] |
| surface | heldout | wrist F/T over vision/state | safe_success | +0.943 * | [+0.907, +0.970] |
| surface | heldout | wrist F/T over vision/state | peak_force | -7.379 * | [-11.032, -3.398] |
| surface | heldout | wrist F/T on top of proprioception | safe_success | -0.013 | [-0.032, +0.000] |
| surface | heldout | wrist F/T on top of proprioception | peak_force | +2.231 * | [+1.870, +2.532] |
| surface | heldout | contact-prediction aux without F/T input | safe_success | -0.007 | [-0.018, +0.000] |
| surface | heldout | contact-prediction aux without F/T input | peak_force | -0.581 * | [-0.889, -0.276] |
| surface | heldout | contact-prediction aux with F/T input | safe_success | -0.007 | [-0.028, +0.012] |
| surface | heldout | contact-prediction aux with F/T input | peak_force | -0.242 * | [-0.474, -0.008] |
| surface | heldout | F/T as input vs. F/T only as training signal | safe_success | -0.007 | [-0.025, +0.010] |
| surface | heldout | F/T as input vs. F/T only as training signal | peak_force | +2.811 * | [+2.557, +3.085] |
| surface | heldout | privileged vs. self-supervised aux labels | safe_success | +0.008 | [-0.005, +0.025] |
| surface | heldout | privileged vs. self-supervised aux labels | peak_force | -0.518 * | [-0.713, -0.323] |

`*` = interval excludes zero.
