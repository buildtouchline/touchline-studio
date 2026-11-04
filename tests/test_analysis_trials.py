import unittest
from touchline.analysis_trials import RECIPES,validate_recipe


class RecipeTests(unittest.TestCase):
    def test_registered_bounded_recipes(self):
        for recipe in RECIPES:self.assertEqual(validate_recipe(recipe),recipe)

    def test_reject_untrusted_parameter_changes(self):
        for change in ({'activation_threshold':float('nan')},{'max_gap_s':10},
                       {'activation_threshold':True},{'command':'echo unsafe'},{'name':'auto_promote'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_recipe({**RECIPES[0],**change})
