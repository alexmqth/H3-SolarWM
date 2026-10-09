# E2 step4 objective comparison

Status: **all_videos_complete_pending_visual_gate**. Visual and action acceptance are not automatic.

| Case | Method | Flow | MP4 frameMAD | MP4 boundaryMAD |
|---|---|---:|---:|---:|
| 118eb5d8b75e1b8ac23a4e9ae77af9a9_D_1015 | zero | — | 19.1331 | 24.1643 |
| 118eb5d8b75e1b8ac23a4e9ae77af9a9_D_1015 | fm_only | — | 19.4456 | 20.8947 |
| 118eb5d8b75e1b8ac23a4e9ae77af9a9_D_1015 | fm_action | — | 19.4573 | 20.8450 |
| dfec8ed3237860eba14d67c089ecd041_A_1080 | zero | — | 12.3277 | 24.0523 |
| dfec8ed3237860eba14d67c089ecd041_A_1080 | fm_only | — | 12.1902 | 24.1816 |
| dfec8ed3237860eba14d67c089ecd041_A_1080 | fm_action | — | 12.2052 | 24.1592 |
| parking_historyA_currentA | zero | +1.645495 | 5.2496 | 9.6773 |
| parking_historyA_currentA | fm_only | +1.657549 | 5.1692 | 9.4134 |
| parking_historyA_currentA | fm_action | +1.678612 | 5.1667 | 9.0660 |
| parking_historyA_currentD | zero | -1.546733 | 4.5379 | 3.0971 |
| parking_historyA_currentD | fm_only | -1.395918 | 4.8200 | 2.9127 |
| parking_historyA_currentD | fm_action | -1.376481 | 4.5579 | 2.9475 |
| parking_historyD_currentA | zero | +0.614104 | 4.3013 | 2.7534 |
| parking_historyD_currentA | fm_only | +0.699057 | 4.3632 | 2.8074 |
| parking_historyD_currentA | fm_action | +0.657532 | 4.5486 | 2.7635 |
| parking_historyD_currentD | zero | -0.673822 | 3.6202 | 3.2072 |
| parking_historyD_currentD | fm_only | -0.667578 | 3.5508 | 3.1529 |
| parking_historyD_currentD | fm_action | -0.616613 | 3.5457 | 3.2759 |

Raw transition-model receipts use pre-encode frames; this table recomputes all comparisons from the same codec to avoid mixing definitions. Parking reference histories are Original-generated, not realGT; natural joint-control cases are realGT but have no pure-lateral flow sign gate.
