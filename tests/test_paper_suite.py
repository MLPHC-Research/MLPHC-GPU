import unittest

from scripts.run_paper_suite import parse_case_specification


class PaperSuiteTest(unittest.TestCase):
    def test_case_specification(self) -> None:
        self.assertEqual(parse_case_specification("1,3,5-7"), [1, 3, 5, 6, 7])

    def test_duplicate_cases_are_removed(self) -> None:
        self.assertEqual(parse_case_specification("1-3,2"), [1, 2, 3])

    def test_invalid_case_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_case_specification("0,1")


if __name__ == "__main__":
    unittest.main()
