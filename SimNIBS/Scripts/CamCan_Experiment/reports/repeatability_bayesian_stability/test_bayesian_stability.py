from __future__ import annotations

import unittest

import numpy as np

import run_bayesian_stability as analysis


class BayesianStabilityTests(unittest.TestCase):
    def test_input_has_ten_complete_paired_participants(self) -> None:
        paired, raw = analysis.load_paired_remesh_cv(analysis.DEFAULT_INPUT)
        self.assertEqual(paired.shape, (10, 2))
        self.assertEqual(tuple(paired.columns), analysis.TARGETS)
        self.assertEqual(len(raw), 40)
        self.assertTrue((paired.to_numpy() > 0).all())

    def test_niw_update_matches_scalar_components(self) -> None:
        values = np.log(
            np.array(
                [
                    [2.0, 2.2],
                    [2.5, 2.4],
                    [3.0, 2.8],
                ]
            )
        )
        prior = analysis.PRIORS[0]
        posterior = analysis.fit_niw(values, prior)
        m0, kappa0, nu0, _ = prior.parameters()
        expected_mean = (kappa0 * m0 + len(values) * values.mean(axis=0)) / (
            kappa0 + len(values)
        )
        np.testing.assert_allclose(posterior.mean, expected_mean)
        self.assertEqual(posterior.kappa, kappa0 + len(values))
        self.assertEqual(posterior.nu, nu0 + len(values))
        np.testing.assert_allclose(posterior.scale, posterior.scale.T)
        self.assertTrue((np.linalg.eigvalsh(posterior.scale) > 0).all())

    def test_predictive_sampling_is_positive_paired_and_reproducible(self) -> None:
        paired, _ = analysis.load_paired_remesh_cv(analysis.DEFAULT_INPUT)
        posterior = analysis.fit_niw(np.log(paired.to_numpy()), analysis.PRIORS[0])
        first = analysis.sample_new_participants(
            posterior, 250, 3, np.random.default_rng(123)
        )
        second = analysis.sample_new_participants(
            posterior, 250, 3, np.random.default_rng(123)
        )
        self.assertEqual(first.shape, (250, 3, 2))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first > 0).all())
        np.testing.assert_allclose(first, second)

    def test_small_end_to_end_stages_have_expected_rows(self) -> None:
        paired, _ = analysis.load_paired_remesh_cv(analysis.DEFAULT_INPUT)
        priors = analysis.PRIORS[:1]
        loo, summary, medians = analysis.run_loo(paired, priors, 2_000, 7)
        self.assertEqual(len(loo), 20)
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(medians), 20)
        expansion, probability = analysis.run_expansion(
            paired,
            priors,
            2_000,
            7,
            12,
            np.array([0.10, 0.25]),
        )
        self.assertEqual(len(expansion), 3 * 2)
        self.assertEqual(len(probability), 3 * 2 * 2)
        current = probability.loc[probability["total_cohort_n"] == 10]
        self.assertTrue(
            (current["probability_absolute_shift_exceeds_delta"] == 0).all()
        )


if __name__ == "__main__":
    unittest.main()
