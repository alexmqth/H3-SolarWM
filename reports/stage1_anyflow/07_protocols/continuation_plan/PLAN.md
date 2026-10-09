# Frozen-time Stage1: duration-only continuation

Wait for frozen16 training and all6 evaluations to finish and its controller PID to exit. Source, shared model, initial image/action base and teacher input hashes are frozen. Only GPU2 is used, serially, reserve20GiB.

If a complete A/D configuration passes A>0,D<0,A-D>1, pause additional training for visual review (this is not Stage1 acceptance). Otherwise restore frozen16 step16 including Adam/RNG, train to32 total updates and evaluate A/D native8. If step32 numeric gate passes, defer further training for visual review. Otherwise restore step32, train to64 total updates and evaluate A/D native4/native8. Step48 is also saved.

All initialization/data/noise/anchor/chunk/routing/shift/loss/LR/target-time policy remain fixed. New output directories retain old results. Checks enforce continued update history and frozen time parameters; evaluation requires39 RGB frames, exact conditioning equality, correct forward/commit counts and full decoding. Visual inspection remains manual. This queue does not start Stage2.

The CUDA resume path is not yet empirically compared to an uninterrupted pretrained run. CPU actual-small-H3 bitwise continuation was verified previously; the queue will exercise the real resume path and stop on any failure, never silently fall back to a weights-only restart.
