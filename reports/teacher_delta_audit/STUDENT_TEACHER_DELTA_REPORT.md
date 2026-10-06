# Causal student vs full-teacher action-delta audit

This read-only audit uses the same generated A rollout state, the same detached causal raw-KV history, the same current chunk and the same five solver sigmas. The causal student is `step_01` from the RGB-dual FP32 gain curve. The teacher is the original bidirectional H3 path with causal QKV and action residual disabled. Only the A/D action conditioning changes between each pair.

| Summary over 15 chunk/sigma states | Value |
|---|---:|
| student A/D delta norm, mean | 69.70 |
| teacher A/D delta norm, mean | 10.68 |
| student / teacher delta norm, mean | 6.89× |
| student / teacher delta norm, median | 7.31× |
| student / teacher delta norm, range | 4.21×–9.91× |
| student-teacher delta cosine, mean | −0.0087 |
| student-teacher delta cosine, median | −0.0115 |
| student-teacher delta cosine, range | −0.2257–+0.1178 |

The student action difference is therefore almost orthogonal to the teacher action difference and much larger in norm. This is a stronger diagnosis than the earlier teacher-only audit: the teacher target is small but finite, while the causal action pathway produces a different action-conditioned score geometry. A/D flow signs can remain superficially correct while the underlying score-field delta is not aligned with H3's action geometry.

The audit is still a single scene, one seed, one A-generated state, one visual adapter and one student checkpoint. It is not a statistical action-accuracy estimate. The exact rows are in `student_teacher_delta_audit.json`; the reproducible script is [`code/causal/audit_student_teacher_delta.py`](../../code/causal/audit_student_teacher_delta.py).

## Decision

Stop extending the per-action gain curve. More gain updates cannot rotate the action delta because gain only rescales the existing residual columns. The next architecture experiment should first expose or adapt the action pathway so the student delta can align with the full-teacher delta, then re-evaluate the same 39-frame gate. A critic/DMD experiment should wait until this action-pathway alignment is measurable; otherwise it would try to correct a representation/topology mismatch with a distribution-level loss.
