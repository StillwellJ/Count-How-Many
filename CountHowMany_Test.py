import unittest
from CountHowMany import counter


class TestObjectCounting(unittest.TestCase):
    def test_multi_car_image(self):
        count = counter("test.jpg")
        self.assertEqual(count, 7)

    def test_no_car_image(self):
        count = counter("test2.jpg")
        self.assertEqual(count, 0)

    def test_one_car_image(self):
        count = counter("test3.jpg")
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
