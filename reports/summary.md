# Results summary

| Approach | Precision | Recall | F1 | False positives / 1000 |
|---|---|---|---|---|
| Rules (Sigma baseline) | 0.95 | 0.82 | 0.88 | 4.3 |
| LLM — prompt v1 | 0.24 | 0.91 | 0.38 | 277.1 |
| LLM — prompt v3 | 0.16 | 1.00 | 0.27 | 515.2 |
| Hybrid (rules ∪ LLM) | 0.16 | 1.00 | 0.27 | 515.2 |



## Full metrics

| Run | P | R | F1 | FP/1000 | TP | FP | FN | TN | Technique acc. |
|---|---|---|---|---|---|---|---|---|---|
| rules | 0.947 | 0.818 | 0.878 | 4.3 | 18 | 1 | 4 | 208 | 100% |
| llm_v1 | 0.238 | 0.909 | 0.377 | 277.1 | 20 | 64 | 2 | 145 | 0% |
| llm_v3 | 0.156 | 1.000 | 0.270 | 515.2 | 22 | 119 | 0 | 90 | 73% |
| hybrid | 0.156 | 1.000 | 0.270 | 515.2 | 22 | 119 | 0 | 90 | 91% |

