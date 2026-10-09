# EXP-001: 73-frame checkpoint review before plan v3

This is a preserved interim record for the four third-chunk runs under the frozen v1 runner and config. It does not assert 90-frame or 124-frame success.

| Path | New RGB 56–72 mean horizontal flow (px/pair) | Boundary gray MAD | Review |
| --- | ---: | ---: | --- |
| AA | +0.347111 | 4.5125 | Direction proxy has expected sign; person remains discernible. |
| DD | −0.878582 | 9.4076 | Direction proxy has expected sign; person remains discernible. |
| AD | −0.972243 | 15.7775 | Direction proxy has expected sign; noticeably larger chunk-boundary difference. |
| DA | −0.175372 | 10.2739 | A direction proxy slightly reversed; first/last 8 pairs both negative. Person approaches camera; bright spot appears later. Do not call this credible left response. |

All four self-history paths have 73 RGB frames. A CPU PyAV audit completely decoded every full video, 17-frame new clip and 2-frame boundary clip as H.264, 832×480, 24 fps with increasing PTS. Source video hashes, endpoint hashes and `published.npy` hashes matched the frozen per-path JSON. The first 56 published RGB frames were byte-identical to their pre-extension arrays. See [video_integrity_audit.json](video_integrity_audit.json), [full73_flow.json](full73_flow.json), [copied videos](videos/) and [all new-frame and boundary sheets](sheets/).

The static review covered the entire new interval and the boundary at RGB56, including full-resolution DA frames 56, 61, 66, 70 and 72. It does not substitute for an independent normal-speed playback judgement. Flow is a motion proxy, not action accuracy. The four paths have diverged histories and cannot be treated as same-state counterfactuals at the third chunk.

The v2 S1 all-path stop criterion was triggered by DA and the fourth chunk was not run under v2. A later plan may narrow the question to sustained AA/DD paths; this interim result must remain visible and cannot be relabelled as passing the original all-path gate.
