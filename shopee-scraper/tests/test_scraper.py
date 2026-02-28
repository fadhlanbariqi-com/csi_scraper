import unittest
from src.retriv import ShopeeScraper

class TestShopeeScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = ShopeeScraper(
            search_term='test',
            max_products=5,
            index_only=False,
            review_limit=10,
            all_star_types=False,
            star_limit_per_type=5,
            chrome_user_data_dir=None
        )

    def test_initialization(self):
        self.assertEqual(self.scraper.search_term, 'test')
        self.assertEqual(self.scraper.max_products, 5)
        self.assertFalse(self.scraper.index_only)
        self.assertEqual(self.scraper.review_limit, 10)

    def test_parse_star_text(self):
        self.assertEqual(self.scraper._parse_star_text('1,2k'), 1200)
        self.assertEqual(self.scraper._parse_star_text('15k'), 15000)
        self.assertEqual(self.scraper._parse_star_text('100'), 100)
        self.assertEqual(self.scraper._parse_star_text('invalid'), 0)

    def test_safe_get(self):
        # This test would require a live environment to run properly
        # Here we can only check if the method exists
        self.assertTrue(hasattr(self.scraper, '_safe_get'))

    def test_retrieve_products(self):
        # This test would require a live environment to run properly
        # Here we can only check if the method exists
        self.assertTrue(hasattr(self.scraper, '_retrieve_products'))

if __name__ == '__main__':
    unittest.main()