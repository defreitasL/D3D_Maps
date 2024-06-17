import unittest
import json
from python_package.api import app
from python_package import config


class TestPythonModule(unittest.TestCase):
    """Example test"""

    def setUp(self) -> None:
        self.client = app.test_client()
        self.client.testing = True

    def test_check_api(self):
        response = self.client.get(config.BASE_URL + "/check-api")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.status, "200 OK")
        self.assertEqual(response.json, "This Flask-Api is up!")
        
    def test_say_hello(self):
        response = self.client.post(config.BASE_URL + "/say-hello", data=json.dumps({"word": "world"}), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.status, "200 OK")
        self.assertEqual(response.json, "Hello, world!")
