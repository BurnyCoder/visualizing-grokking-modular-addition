| ID | Quantity | Paper / ARENA | Ours | Match | Source | Note |
|---|---|---|---|---|---|---|
| R1 | loss at epoch 40k (all 12,769 pairs, 113-way) | 2.41e-07 | 2.412e-07 | yes | p.6–7 | The paper calls it test loss; it matches the all-pairs loss (test-only is 3.02e-7). |
| R2 | key frequencies | [14, 35, 41, 42, 52] | [14, 35, 41, 42, 52] | yes | p.4 |  |
| R3 | non-negligible W_E frequencies | 6 | 6 | yes | p.4 | threshold 10 % of max; the 6th is k=31 |
| R4 | W_L residual after 10-direction approximation | <0.55 % | 0.005517 | yes | p.6 | squared-norm ratio 0.552 % at step 40,001 (match: < 0.6 %) and 0.520 % at the last checkpoint (step 49,901); the norm ratio, 7.4 %, does not match the paper's wording (Frobenius norm under 0.55 % of ‖W_L‖) |
| R5 | neurons with >85 % FVE at one frequency | 433 | 433 | yes | p.6 |  |
| R6 | neurons in the k=14 cluster | 44 | 44 | yes | p.7 |  |
| R7 | Table 1 FVE of u_k/v_k read-offs (mean of 10) | 0.9624 | 0.9635 | yes | p.6 Table 1 | paper's ten values average 96.24 %; per-row values in details.table1 |
| R8a | logit fit Σα_k cos(w_k(a+b−c)) FVE | 0.95 | 0.9614 | yes | p.6 |  |
| R8b | loss of the fitted logits | 4.7e-08 | 4.395e-08 | yes | p.6 |  |
| R9a | loss keeping only the 40 key 2D components | 7.24e-08 | 7.236e-08 | yes | p.7 |  |
| R9b | MLP projected onto 10 W_L directions | 1.19e-07 | 2.377e-07 | **no** | p.8 | closest variant shown; all variants in details.wl_projection |
| R9c | MLP projected onto the nullspace | 5.27 | 4.804 | **no** | p.8 | closest variant (MLP-only, bias-corrected) shown |
| R10a | clustered neurons replaced by degree-2 polynomial | 2.48e-07 | 2.482e-07 | yes | p.7 |  |
| R10b | neurons restricted to cos/sin(w_k(a+b)) terms | 5.54e-08 | 1.121e-07 | **no** | p.7 | per-neuron restriction gives this value; the paper's 5.54e-8 matches the restricted-LOGITS loss (R19a, 5.547e-08) within 0.2 %, so the paper likely restricted logits |
| R11a | MLP skip connection zero-ablated | 9.12e-07 | 8.877e-07 | yes | p.13 |  |
| R11b | MLP skip connection mean-ablated | 7.25e-07 | 7.248e-07 | yes | p.13 |  |
| R11c | attention heads zero-ablated | 24.3 | 24.27 | yes | p.13 |  |
| R11d | skip around attention zero-ablated | 19.1 | 1.086 | **no** | p.13 | not reproduced: our ablation sets resid_mid = attention output (the paper does not specify its hook) |
| R12 | mean attention from '=' to '=' (per head) | 0.1–0.4 % | [0.004418, 0.001111, 0.004427, 0.001177] | yes | p.13 |  |
| R13a | linear fit of the attention softmax (variance explained) | 0.975 | 0.9756 | yes | p.16 |  |
| R13b | loss with softmax replaced by its linear fit | 2.12e-07 | 2.124e-07 | yes | p.16 |  |
| R14 | head frequencies (Table 2) | [35, 42, 52, 42] | [35, 42, 52, 42] | yes | p.15 Table 2 |  |
| R16 | ARENA excluded loss per key freq | [0.000282, 0.000633, 0.00161, 0.07195, 0.03288] | [0.000282, 0.0006332, 0.00161, 0.07195, 0.03288] | yes | ARENA tests.test_excl_loss |  |
| R17 | ARENA cluster sizes 14/35/41/42/52/always | [44, 93, 145, 87, 64, 79] | [44, 93, 145, 87, 64, 79] | yes | ARENA 2-circuit-and-feature-analysis |  |
| R18 | ARENA ReLU regression r² (with / without quadratic) | [0.966, 0.849] | [0.9659, 0.8494] | yes | ARENA |  |
| R19a | ARENA trig-only (restricted) logits loss | 5.547e-08 | 5.547e-08 | yes | ARENA |  |
| R19b | ARENA key-frequency neuron projection loss (with / without always-firing) | [2.482e-07, 1.179e-06] | [2.482e-07, 1.179e-06] | yes | ARENA |  |
| R15 | SSW at the paper's phase boundaries 1.4k / 9.4k / 14k (Fig. 7) | falls; steepest in cleanup | [3373, 2001, 995.5] | yes | p.9 Fig. 7 | qualitative; fall per step 0.171 (circuit formation) vs 0.219 (cleanup) |
| R20a | total SSW at epoch 40k and at the end of training | [955, 942] | [955.4, 941.8] | yes | ARENA §3 Analysis During Training | last checkpoint: step 49901 |
| R20b | attention to '=' returns after ~43k (attention at 40k → 45k) | attends to '=' again | [[0.004418, 0.001111, 0.004427, 0.001177], [0.1133, 0.0008202, 0.09146, 0.0008718]] | yes | ARENA §3 Analysis During Training | 2 heads rise ≥ 5× to > 5 %; 2 such heads is our match criterion (ARENA gives no head count) |
