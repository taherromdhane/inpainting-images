import unittest
import numpy as np
np.set_printoptions(precision=3)

from inpaint.Inpainter import Inpainter

class TestInpainter(unittest.TestCase):

    def setUp(self) :
        """
        Initializing the inpainter with most common parameters for testing
        """
        self.patch_size = 3
        self.local_radius = None
        self.data_significance = 0
        self.alpha = 1
        self.threshold = None
        self.inpainter = Inpainter(self.patch_size, self.local_radius, self.data_significance, self.alpha, self.threshold)

        image = np.random.rand(6, 6, 3) 

        mask = np.array([[1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 1, 1],
                        [1, 1, 0, 0, 1, 1],
                        [1, 0, 0, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        ])
        self.center = (2, 1)

        self.inpainter.image = image
        self.inpainter.mask = mask
        self.inpainter.confidence = np.copy(self.inpainter.mask)

    # Testing Border Pixel
    def test_border_px_no_offset(self) :
        """
        Testing the border extracting method from the mask, with no offset
        """
        mask = np.array([[1, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0],
                         [1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 1],
                         [1, 1, 1, 1, 1]
                        ])

        self.inpainter.mask = mask
        self.inpainter.offset = 0
        self.inpainter._getBorderPx()

        result_no_offset = {(0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 3), \
                            (2, 0), (3, 0), (3, 1), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4)}
        self.assertEqual(self.inpainter.border_pxls, result_no_offset)

    def test_border_px_with_offset(self) :
        """
        Testing the border extracting method from the mask, with offset
        (not getting pixels whose patches go over mask border)
        """
        mask = np.array([[1, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0],
                         [1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 1],
                         [1, 1, 1, 1, 1]
                        ])

        self.inpainter.mask = mask
        self.inpainter._getBorderPx()

        result_with_offset = {(1, 1), (1, 3), (3, 1)}
        self.assertEqual(self.inpainter.border_pxls, result_with_offset)

    # Testing getting max priority
    ## Testing confidence term
    def test_confidence_term(self) :
        """
        Testing the confidence term of a patch 
        """
        mask = np.array([[1, 1, 1, 1],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]
                        ])
        self.inpainter.mask = mask
        self.inpainter.confidence = np.copy(self.inpainter.mask)

        center = (1, 1)
        Cp = self.inpainter._patchConfidence(center)
        self.assertAlmostEqual(Cp, 0.4444444444444444)

    ## Testing data term
    def test_normal_mask(self) :
        """
        Testing the normal vector of all pixels of the mask
        """
        unit_normal = self.inpainter._calcNormalMask()
        self.assertEqual(unit_normal.shape[:2], self.inpainter.mask.shape)
        self.assertEqual(unit_normal.shape[2], 2)

        norm = np.sqrt(unit_normal[:, :, 0] ** 2 + unit_normal[:, :, 1] ** 2)
        self.assertTrue(np.all(np.logical_or(np.isclose(norm, 1), np.isclose(norm, 0))))

    def test_gradient_mask(self):
        """
        Testing the gradient vector for all pixels of the image
        """
        gradient, gradient_val = self.inpainter._calcGradientMask()
        self.assertEqual(gradient.shape[1:], self.inpainter.mask.shape)
        self.assertEqual(gradient_val.shape, self.inpainter.mask.shape)

    def test_normal_vector(self) :
        """
        Testing the normal vector for a pixel
        """
        self.inpainter._prepareData()
        normal_vector = self.inpainter._getNormalPatch(self.center)
        self.assertEqual(normal_vector.size, 2)

    def test_gradient_vector(self) :
        """
        Testing the gradient vector for all pixels of the image
        """
        self.inpainter._prepareData()
        max_gradient = self.inpainter._getGradientPatch(self.center)

        self.assertEqual(max_gradient.size, 2)

    def test_data_term(self) :
        """
        Testing the data term for the selected pixel
        """        
        self.inpainter._prepareData()
        data = self.inpainter._patchData(self.center)
        
        self.assertIsInstance(data, float)

    def test_max_priority(self) :
        """
        Testing the max priority
        """ 
        self.inpainter._getBorderPx()
        max_pixel, Cp = self.inpainter._getMaxPriority()
        self.assertEqual(len(max_pixel), 2)
        self.assertIsInstance(Cp, float)

    ## Testing max priority