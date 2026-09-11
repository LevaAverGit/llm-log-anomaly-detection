# Cross-model comparison

| Model | F1 (v1) | F1 (v3) |
|---|---|---|
| Rules (Sigma baseline) * | 0.88 | 0.88 |
| gemma3 | 0.38 | 0.27 |
| Qwen2.5 7B | 0.76 | 0.70 |
| DeepSeek-R1 7B | 0.55 | 0.64 |
| GigaChat-2 (Sber) | 0.60 | 0.68 |

_\* Rules are deterministic and do not use a prompt; the same F1 is shown in every column for reference._


## Per-model error breakdown

- **gemma3**
  - **v1**: recall 0.91 (20/22), precision 0.24, FP 277/1000. Under-detects: T1562.007 (1), T1595.002 (1). Over-detects: linux_auth (54), cloud_audit (7), windows_security (3).
  - **v3**: recall 1.00 (22/22), precision 0.16, FP 515/1000. Under-detects: no incident classes missed. Over-detects: linux_auth (50), nginx_access (35), cloud_audit (21).
- **Qwen2.5 7B**
  - **v1**: recall 0.77 (17/22), precision 0.74, FP 26/1000. Under-detects: T1059.001 (1), T1078 (1), T1110 (1). Over-detects: linux_auth (5), windows_security (1).
  - **v3**: recall 0.86 (19/22), precision 0.59, FP 56/1000. Under-detects: T1078 (1), T1098 (1), T1190 (1). Over-detects: linux_auth (10), cloud_audit (3).
- **DeepSeek-R1 7B**
  - **v1**: recall 0.41 (9/22), precision 0.82, FP 9/1000. Under-detects: T1190 (3), T1110.001 (2), T1562.007 (2). Over-detects: cloud_audit (1), linux_auth (1).
  - **v3**: recall 0.73 (16/22), precision 0.57, FP 52/1000. Under-detects: T1190 (2), T1078 (1), T1098 (1). Over-detects: linux_auth (11), windows_security (1).
- **GigaChat-2 (Sber)**
  - **v1**: recall 0.64 (14/22), precision 0.56, FP 48/1000. Under-detects: T1595.002 (2), T1059.001 (1), T1078 (1). Over-detects: linux_auth (10), cloud_audit (1).
  - **v3**: recall 0.86 (19/22), precision 0.56, FP 65/1000. Under-detects: T1078 (1), T1190 (1), T1595.002 (1). Over-detects: linux_auth (10), cloud_audit (4), windows_security (1).

