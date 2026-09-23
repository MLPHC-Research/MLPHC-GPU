import unittest

from mlphc_gpu import HCLPSOConfig, PAPER_CONFIG


class PaperDefaultsTest(unittest.TestCase):
    def test_paper_configuration(self) -> None:
        self.assertEqual(PAPER_CONFIG.hidden_neurons, 8)
        self.assertEqual(PAPER_CONFIG.rule_dimension, 40)
        self.assertEqual(PAPER_CONFIG.population_size, 20)
        self.assertEqual(PAPER_CONFIG.parameter_scale, 2.0)
        self.assertEqual(PAPER_CONFIG.max_evaluations(36), 1800)

    def test_hclpso_split_for_paper_population(self) -> None:
        config = HCLPSOConfig(population_size=PAPER_CONFIG.population_size)
        exploration = int(
            config.population_size * config.exploration_fraction + 0.5
        )
        self.assertEqual(exploration, 8)
        self.assertEqual(config.population_size - exploration, 12)


if __name__ == "__main__":
    unittest.main()
