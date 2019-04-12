import io
import sys
import unittest

import utils

class TestUtils(unittest.TestCase):
    def test_std_flush(self):
        """Redirect standard output to a temporary buffer, call std_flush, and test the content
        written to that temporary buffer."""
        red_stdout = io.StringIO()
        sys.stdout = red_stdout
        # No parameters.
        utils.std_flush()
        self.assertEqual(red_stdout.getvalue(), "\n")
        # Parameters of different types.
        utils.std_flush(42, "Hello", [1, 2, 3])
        self.assertEqual(red_stdout.getvalue(), "\n42 Hello [1, 2, 3]\n")

if __name__ == "__main__":
    unittest.main()
