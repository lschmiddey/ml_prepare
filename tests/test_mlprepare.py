from prep_funcs import *
import unittest

class TestCases(unittest.TestCase):
    def test_ifnone(self):
        a, b = None, 5
        expected_result = 5
        self.assertEqual(ifnone(a,b), expected_result)
        
        a,b = 10,5
        expected_result = 10
        self.assertEqual(ifnone(a,b), expected_result)
        
if __name__ == '__main__':
    unittest.main()