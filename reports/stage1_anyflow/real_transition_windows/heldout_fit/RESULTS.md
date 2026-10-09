# Held-out observed-transition fit: 0 vs 4 updates

48 matched-state forwards; two validation transitions, four sigmas, positive/swapped action, three parameter banks. No coefficient selection or retraining uses these values.

| Sigma | Bank | Positive FM | Swapped-action FM | Gap (neg − pos) | Action delta RMS | Correct order |
|---|---|---:|---:|---:|---:|---:|
| all | zero | 0.25976668 | 0.25974831 | -0.00001837 | 0.02027782 | 5/8 |
| all | fm_only | 0.25969061 | 0.25967603 | -0.00001458 | 0.02022031 | 4/8 |
| all | fm_action | 0.25969900 | 0.25967836 | -0.00002065 | 0.02018351 | 4/8 |
| 0.85 | zero | 0.14874313 | 0.14857883 | -0.00016430 | 0.02042481 | 1/2 |
| 0.85 | fm_only | 0.14859660 | 0.14841907 | -0.00017753 | 0.02036323 | 0/2 |
| 0.85 | fm_action | 0.14860538 | 0.14843677 | -0.00016861 | 0.02025207 | 0/2 |
| 0.55 | zero | 0.17607000 | 0.17612550 | +0.00005550 | 0.01520384 | 2/2 |
| 0.55 | fm_only | 0.17605571 | 0.17609868 | +0.00004297 | 0.01516922 | 2/2 |
| 0.55 | fm_action | 0.17605826 | 0.17610034 | +0.00004209 | 0.01516997 | 2/2 |
| 0.25 | zero | 0.28089511 | 0.28085999 | -0.00003512 | 0.01866819 | 0/2 |
| 0.25 | fm_only | 0.28085081 | 0.28082840 | -0.00002241 | 0.01862257 | 0/2 |
| 0.25 | fm_action | 0.28085983 | 0.28083943 | -0.00002040 | 0.01860842 | 0/2 |
| 0.1 | zero | 0.43335849 | 0.43342893 | +0.00007044 | 0.02681445 | 2/2 |
| 0.1 | fm_only | 0.43325931 | 0.43335795 | +0.00009865 | 0.02672621 | 2/2 |
| 0.1 | fm_action | 0.43327254 | 0.43333688 | +0.00006434 | 0.02670358 | 2/2 |

## Interpretation

Both arms reduce mean positive FM by less than 0.03% after four updates. FM+action does not improve the mean action ranking over FM-only: correct ordering is 4/8 in both, versus 5/8 at initialization. Its mean action delta RMS also does not increase. This is no evidence of added benefit from this short ranking-loss trial.

The high-noise sigma0.85 case dominates the negative average gap, while signs vary across state and sigma. Larger low-noise absolute FM alone is not a controlled demonstration that the training noise distribution caused the video artifacts. Do not change sigma weighting or lambda based on this validation set.

A lower swapped-action error does not prove a swapped-action video is correct: the target is the one actually observed transition. The two scenes contain joint character and camera controls, and there are no observed alternate-action videos. This diagnostic must be considered together with pure-action parking generation and full-frame structure review.

Four optimizer updates and two held-out states cannot establish that action-consequence supervision is impossible or that more training would necessarily help. They do not justify promoting this checkpoint to AnyFlow or Stage2.
