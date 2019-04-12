import datetime
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

    def test_readable_time(self):
        # Test standard format.
        self.assertTrue(datetime.datetime.strptime(utils.readable_time(), "%M:%S").minute in
                [datetime.datetime.now().minute, (datetime.datetime.now().minute - 1) % 60])
        self.assertTrue(datetime.datetime.strptime(utils.readable_time(), "%M:%S").second in
                [datetime.datetime.now().second, (datetime.datetime.now().second - 1) % 60])
        # Test format parameter.
        self.assertTrue(
            datetime.datetime.strptime(utils.readable_time("%H:%M:%S"), "%H:%M:%S").hour in
            [datetime.datetime.now().hour, (datetime.datetime.now().hour - 1) % 24]
        )
        self.assertTrue(
            datetime.datetime.strptime(utils.readable_time("%H:%M:%S"), "%H:%M:%S").minute in
            [datetime.datetime.now().minute, (datetime.datetime.now().minute - 1) % 60]
        )
        self.assertTrue(
            datetime.datetime.strptime(utils.readable_time("%H:%M:%S"), "%H:%M:%S").second in
            [datetime.datetime.now().second, (datetime.datetime.now().second - 1) % 60]
        )

if __name__ == "__main__":
    unittest.main()
