import numpy as np
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
import math

class Inpainter :

    def _normalize_image(self) :
        # Utility method to normalize the image before running
        # the inpainting algorithm

        return self.image/np.max(self.image)

    def _getPatch(self, array, center) :
        # Utility method to get patch of image or mask or whatever

        i, j = center
        
        try :
            if len(array.shape) == 2 :
                return array[i - self.offset : i + self.offset + 1, j - self.offset : j + self.offset + 1]
            
            if len(array.shape) == 3 :
                return array[i - self.offset : i + self.offset + 1, j - self.offset : j + self.offset + 1, :]

            raise ValueError("Not a valid call, must be array either in 2d format or image in RGB/RGBA format")

        except IndexError :
            print("One of the indices is out of range")
        

    # Methods for getting border pixels
    def _isBorderPixel(self, n, m) :
        # Utility method that takes a pixel position (n, m) and a mask as input and returns
        # whether the pixel in that position is a border pixel or not
        
        if self.mask[n][m] == 0 :
            return False
        
        for i in range(-1, 2) :
            for j in range(-1, 2) :
                if i == 0 and j == 0 or n+i<0 or m+j<0:
                    continue
                try :
                    if self.mask[n + i][m + j] == 0 :
                        return True
                except IndexError :
                    continue
                    
        return False

    def _getBorderPx(self) :
        # Utility method to get all the border pixels in the image
        # excluding of course those who have a patch that partially goes 
        # over the border

        self.border_pxls = set()

        upper_i = self.mask.shape[0] - self.offset - 1
        lower_i = self.offset
        upper_j = self.mask.shape[1] - self.offset - 1
        lower_j = self.offset
        
        for i in range(lower_i, upper_i + 1) :
            for j in range(lower_j, upper_j + 1) :
                if self._isBorderPixel(i, j) :
                    self.border_pxls.add((i, j))


    # Methods to calculate the maximum priority

    ## Calculate the patch confidence
    def _patchConfidence(self, center) :
        # Utility method to calculate the confidence of a chosen patch

        i, j = center
        return np.sum(self._getPatch(self.confidence, center)) / self.patch_size**2
    
    ## Calculate the patch data for the whole mask
    def _calcNormalMask(self) :
        # Utility method to calculate the normal vector to each pixel in the mask

        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[.25, .5, .25], [0, 0, 0], [-.25, -.5, -.25]])
        
        x_normal = convolve(self.mask.astype(float), x_kernel)
        y_normal = convolve(self.mask.astype(float), y_kernel)
        
        normal = np.dstack((x_normal, y_normal))
        
        norm = np.sqrt(normal[:, :, 0]**2 + normal[:, :, 1]**2)
        norm[norm == 0] = 1

        unit_normal = -normal / np.expand_dims(norm, axis=2)
        return unit_normal
        
    def _calcGradientMask(self):
        # Utility method to calculate the gradient vector to each pixel in the mask
        
        grey_image = rgb2gray(self.image)
        grey_image[self.mask == 0] = None
        
        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)

        return gradient, gradient_val
        
    def _prepareData(self) :

        self.normal_mask = self._calcNormalMask()
        self.gradient, self.gradient_val = self._calcGradientMask()

    ## Methods to get the normal and gradient vectors to a patch
    def _getNormalPatch(self, center):
        # Utility method to calculate the normal vector to a chosen patch

        i, j = center
        return self.normal_mask[i, j]

    def _getGradientPatch(self, center) :
        # Utility method to get the maximum norm of a gradient vector 
        # to the pixels of a chosen patch
        
        max_gradient = np.zeros([2])
        i, j = center
        
        patch_y_gradient = self._getPatch(self.gradient[0], center)
        patch_x_gradient = self._getPatch(self.gradient[1], center)
        patch_gradient_val = self._getPatch(self.gradient_val, center)

        patch_max_pos = np.unravel_index(
            patch_gradient_val.argmax(),
            patch_gradient_val.shape
        )

        max_gradient[0] = patch_y_gradient[patch_max_pos]
        max_gradient[1] = patch_x_gradient[patch_max_pos]

        return max_gradient
    
    def _patchData(self, center) :
        
        i, j = center

        normal_vector = self._getNormalPatch(center)
        max_gradient_vector = self._getGradientPatch(center)
        
        data = np.abs(np.dot(normal_vector, max_gradient_vector.T)) / self.alpha
        
        return data
    
    ## Actually calculate the maximum priority 
    def _getMaxPriority(self) :
        
        Pp, Cp = 0, 0
        max_pixel = (0, 0)
        
        self._prepareData()
        
        for pixel in self.border_pxls :
            
            n, m = pixel
            
            # Check for going over border already performed in _getBorderPx
            # if n - offset < 0 or n + offset + 1 > image.shape[0] or m - offset < 0 or m + offset + 1 > image.shape[1] :
            #     continue
            
            current_Cp = self._patchConfidence(pixel)
            current_Dp = self._patchData(pixel)
            # current_Dp = 1
            
            current_Pp = current_Cp * (current_Dp ** self.data_significance) # Pp to change into matrix
            
            if current_Pp >= Pp :
                Pp = current_Pp
                Cp = current_Cp
                max_pixel = pixel

        return max_pixel, Cp


    # Get the optimal patch to use for the filling

    def _getSearchBoundaries(self, target_pixel) :
        
        n, m = target_pixel

        if self.local_radius :
            upper_i = min(n + self.local_radius, self.image.shape[0] - self.offset - 1)
            lower_i = max(n - self.local_radius, self.offset)
            upper_j = min(m + self.local_radius, self.image.shape[1] - self.offset - 1)
            lower_j = max(m - self.local_radius, self.offset)
        else :
            upper_i = self.image.shape[0] - self.offset - 1
            lower_i = self.offset
            upper_j = self.image.shape[1] - self.offset - 1
            lower_j = self.offset
        
        return upper_i, lower_i, upper_j, lower_j

    def _patchDistance(self, target_patch, candidate_patch, mask_patch) :

        mask_patch = np.expand_dims(mask_patch, axis=2)
        
        return np.sum(((target_patch - candidate_patch) * mask_patch) ** 2) / np.sum(mask_patch)
    
    def _getOptimalPatch(self, target_pixel) :
        
        upper_i, lower_i, upper_j, lower_j = self._getSearchBoundaries(target_pixel)

        n, m = target_pixel
        
        optimal_patch = (0, 0)
        optimal_distance = 1e9
        target_patch = self._getPatch(self.image, target_pixel)           
        mask_patch = self._getPatch(self.mask, target_pixel)   
        
        threshold_targets = []

        for i in range(lower_i, upper_i + 1) :
            for j in range(lower_j, upper_j + 1) :
                # if candidate patch is in part in mask then skip            
                if np.any(self._getPatch(mask, (i, j)) == 0) :
                    continue
                
                d_center = np.sqrt(np.sum((self.image[n, m, :] - self.image[i, j, :]) ** 2))
                if self.threshold == None or d_center < self.threshold :
                    threshold_targets.append((i, j, d_center))

        threshold_targets.sort(key = lambda x : x[2])
        threshold_targets = threshold_targets[:int(math.sqrt(len(threshold_targets)))]

        for i, j, d_center in threshold_targets :
            candidate_patch = self._getPatch(self.image, (i, j))
            current_distance = self._patchDistance(target_patch, candidate_patch, mask_patch)
            
            if current_distance < optimal_distance :
                optimal_patch = (i, j)
                optimal_distance = current_distance
        
        return optimal_patch

    def _updateConfidence(self, Cp, target_pixel) :   
        
        i, j = target_pixel

        self.confidence[i - self.offset : i + self.offset + 1, j - self.offset : j + self.offset + 1] = \
                self._getPatch(self.confidence, target_pixel)+ (1 - self._getPatch(self.mask, target_pixel)) * Cp


    def _fillPatch(self, target_pixel, opt_patch) :
        
        n, m = target_pixel
        i, j = opt_patch
        offset = self.offset
        
        un, dn, lm, rm = n - offset, n + offset + 1, m - offset, m + offset + 1
        ui, di, lj, rj = i - offset, i + offset + 1, j - offset, j + offset + 1
        
        mask_patch = self._getPatch(self.mask, target_pixel)
        mask_patch = np.expand_dims(mask_patch, axis=2)
        
        self.image[un: dn, lm: rm, :] = self._getPatch(self.image, target_pixel) * mask_patch + \
                                        self._getPatch(self.image, opt_patch) * (1 - mask_patch)
        
        self.mask[un: dn, lm: rm] = 1


    def __init__(self, patch_size, local_radius, data_significance = 0, alpha=1, threshold = None) :

        # assert patch_size is an odd number
        if patch_size%2 == 0 :
            raise ValueError("Patch size must be an odd number for this algorithm !")
        self.patch_size = patch_size
        self.offset = self.patch_size // 2

        self.local_radius = local_radius
        self.data_significance = data_significance
        self.alpha = alpha
        self.threshold = threshold

        

    def inpaint(self, image, mask) :
        
        self.image = image
        self.mask = mask
        if mask.shape[:2] != image.shape[:2] :
            raise ValueError("Mask and image must be of the same shape")

        
        self.confidence = np.copy(self.mask)
        self._normalize_image()
        
        start_zeros = np.sum((1 - self.mask))

        # change to identify border, then calculate priorities
        while True :

            self._getBorderPx(patch_size)
            if len(self.border_pxls) == 0 :
                break
                
            target_pixel, Cp = self._getMaxPriority()

            opt_patch = self._getOptimalPatch(target_pixel)

            self._updateConfidence(target_pixel)

            self._fillPatch(target_pixel, opt_patch)
            
            print("Almost there ! ===> {:.1f}/{}".format((1 - np.sum((1 - self.mask)) / start_zeros) * 100, 100), sep='\n')
            
        return self.image

    # def inpaint_one_iteration(self, image, mask) :
