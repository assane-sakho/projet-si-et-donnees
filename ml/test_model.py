"""
Unit tests for the calculator library
"""

from models import guess_cloth_category_model, guess_cloth_color_model, guess_cloth_brand_model

class TestModel:

    def test_category(self):
        assert 4 == guess_cloth_category_model.predict_category()