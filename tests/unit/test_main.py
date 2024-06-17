import unittest

from python_package import main

class TestPythonModule(unittest.TestCase):

    def test_python_module(self):
        response = main.hello_world("world")
        self.assertIsNotNone(response)
        self.assertEqual(response, "Hello, world!")
