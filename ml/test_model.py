"""
Unit tests for the calculator library
"""

from models import guess_cloth_category_model
from imageio import imread
import unittest

class TestModel:

    #def test_category(self):
    #    assert 4 == guess_cloth_category_model.predict_category()

    def vrai_pantalon(self):
        image = imread('https://www.millet.fr/media/catalog/product/cache/5/image/658x/9df78eab33525d08d6e5fb8d27136e95/m/i/miv8656-0247-pantalon-homme-noir-trilogy-signature-chino-pt-m_3.jpg')
        prediction_image=guess_cloth_category_model.predict(image)
        if(prediction_image=='pantalon'):
            self.assertTrue("Vrai pantalon: ",prediction_image)
        else:
            self.assertTrue("Vrai pantalon: ",prediction_image)
       

#    def faux_pantalon(self):
#        image = imread('https://lp2.hm.com/hmgoepprod?set=quality%5B79%5D%2Csource%5B%2F1e%2F86%2F1e86c77fb86afc19daad9acb5b39470e7bc5ca1f.jpg%5D%2Corigin%5Bdam%5D%2Ccategory%5Bmen_tshirtstanks_shortsleeve%5D%2Ctype%5BDESCRIPTIVESTILLLIFE%5D%2Cres%5Bm%5D%2Chmver%5B2%5D&call=url[file:/product/main]')
#        prediction_image=guess_cloth_category_model.predict(image)
#        if(prediction_image!='pantalon'):
#            self.assertFalse("Faux pantalon: ",prediction_image)
#        else:
#            self.assertTrue("Faux pantalon: ",prediction_image)

if __name__ == '__main__':
    unittest.vrai_pantalon()
