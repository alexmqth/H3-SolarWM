import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from causal.real_transition_data import A, S, bounded_keys9, swap_current_lateral, window_eligibility


class TestRealTransitionData(unittest.TestCase):
    def test_future_camera_leak_is_detected_and_bounded(self):
        p = np.zeros((37,17), dtype=np.float32)
        p[:,A.KEY_COLS.index('J')] = 1
        future = p.copy(); future[36,A.NUM_KEYS+1] = 3
        self.assertNotEqual(S.keys9(p)[35,-1], S.keys9(future)[35,-1])
        np.testing.assert_array_equal(bounded_keys9(p)[:36], bounded_keys9(future)[:36])

    def test_prefixes_are_immutable_with_future_action_perturbation(self):
        rng = np.random.default_rng(13)
        m = rng.normal(size=(124,17)).astype(np.float32)
        m[:,:11] = rng.integers(0,2,size=(124,11))
        expected = bounded_keys9(A.bin_to_latent(m,37))
        for stop in (12,24,36):
            cutoff = A.frame_spans(37)[stop-1][1]
            future = m.copy(); future[cutoff:] *= -100
            np.testing.assert_array_equal(expected[:stop], bounded_keys9(A.bin_to_latent(future,37))[:stop])
            prefix = A.bin_to_latent(m[:cutoff],stop)
            np.testing.assert_array_equal(expected[:stop], bounded_keys9(prefix))

    def test_translation_never_enters_conditioning(self):
        p = np.zeros((37,17),dtype=np.float32); p[:,7] = 1
        q = p.copy(); q[:,14:] = 999
        np.testing.assert_array_equal(bounded_keys9(p), bounded_keys9(q))

    def test_joint_controls_persist_in_negative(self):
        k = np.zeros((37,9),dtype=np.float32)
        for name in ('W','A','J','F'): k[:,S.KEYS9.index(name)] = 1
        n = swap_current_lateral(k,12,24)
        np.testing.assert_array_equal(n[:12],k[:12]); np.testing.assert_array_equal(n[24:],k[24:])
        for name in ('W','S','I','J','K','L','F'):
            np.testing.assert_array_equal(n[:,S.KEYS9.index(name)],k[:,S.KEYS9.index(name)])
        self.assertTrue(np.all(n[12:24,S.KEYS9.index('D')]==1))
        self.assertIn('walks forward and strafes right, camera pans left sharply',S.annotate_from_keys9(n)[12])
        np.testing.assert_array_equal(swap_current_lateral(n,12,24),k)

    def test_reject_ambiguous_or_unrepresentable_labels(self):
        m = np.zeros((124,17),dtype=np.float32); m[:,1] = 1
        self.assertTrue(window_eligibility(m,12,24)['ranking_eligible'])
        m[40,3] = 1
        self.assertIn('opposing_keys_within_latent_bin',window_eligibility(m,12,24)['rejection_reasons'])
        m[:,3] = 0; m[40,4] = 1
        self.assertIn('unsupported_key_in_current_window',window_eligibility(m,12,24)['rejection_reasons'])
        self.assertTrue(window_eligibility(m,24,36)['ranking_eligible'])


if __name__ == '__main__': unittest.main()
