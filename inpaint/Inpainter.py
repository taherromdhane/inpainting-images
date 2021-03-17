import numpy as np
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
import math
import time

class Inpainter :
    
    def normalize(self) :
        # Utility method to normalize the image before running
        # the inpainting algorithm

        self.image = self.image/np.max(self.image)

    def getPatch(self, array, center) :
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

    def isBorderPixel(self, n, m) :
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

    def getBorderPx(self) :
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
                if self.isBorderPixel(i, j) :
                    self.border_pxls.add((i, j))

    # Methods to calculate the maximum priority

    ## Calculate the patch confidence

    def patchConfidence(self, center) :
        # Utility method to calculate the confidence of a chosen patch
        
        return np.sum(self.getPatch(self.confidence, center)) / self.patch_size**2
        
    ## Calculate the patch data for the whole mask
    def calcNormalMatrix(self):
        # Utility method to calculate the normal vector to each pixel in the mask
        
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[.25, .5, .25], [0, 0, 0], [-.25, -.5, -.25]])
        
        x_normal = convolve(self.mask.astype(float), x_kernel)
        y_normal = convolve(self.mask.astype(float), y_kernel)
        
        normal = np.dstack((x_normal, y_normal))

        norm = np.sqrt(normal[:, :, 0]**2 + normal[:, :, 1]**2)
        norm[norm == 0] = 1

        unit_normal = -normal / np.expand_dims(norm, axis=2)
        self.normal_mask = unit_normal
        
    def getNormalPatch(self, center):
        # Utility method to calculate the normal vector to a chosen patch

        i, j = center
        return self.normal_mask[i, j]

    ## Methods to get the normal and gradient vectors to a patch
    def calcGradientMask(self):
        # Utility method to calculate the gradient vector to each pixel in the mask
        
        grey_image = rgb2gray(self.image)
        grey_image[self.mask == 0] = None

        self.gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        self.gradient_val = np.sqrt(self.gradient[0]**2 + self.gradient[1]**2)

    def getGradientPatch(self, center) :
        # Utility method to get the maximum norm of a gradient vector 
        # to the pixels of a chosen patch
        
        max_gradient = np.zeros([2])
        
        patch_y_gradient = self.getPatch(self.gradient[0], center)
        patch_x_gradient = self.getPatch(self.gradient[1], center)
        patch_gradient_val = self.getPatch(self.gradient_val, center)

        patch_max_pos = np.unravel_index(
            patch_gradient_val.argmax(),
            patch_gradient_val.shape
        )

        max_gradient[0] = patch_y_gradient[patch_max_pos]
        max_gradient[1] = patch_x_gradient[patch_max_pos]

        return max_gradient
        
    def prepareDataUtils(self) :
        
        self.calcNormalMatrix()
        self.calcGradientMask()
    
    def patchData(self, center) :
        # Method that calculates the data term for a given patch
        
        normal_vector = self.getNormalPatch(center)
        max_gradient = self.getGradientPatch(center)
        
        data = np.abs(np.dot(normal_vector, max_gradient.T)) / self.alpha
        
        return data
    

    ## Actually calculate the maximum priority 
    def getMaxPriority(self) :
        
        Pp, Cp = 0, 0
        max_pixel = (0, 0)
        # print("border", len(self.border_pxls))
        self.prepareDataUtils()
        
        for pixel in self.border_pxls :
            
            n, m = pixel
            
            # if n - offset < 0 or n + offset + 1 > image.shape[0] or m - offset < 0 or m + offset + 1 > image.shape[1] :
            #     continue
            
            current_Cp = self.patchConfidence(pixel)
            current_Dp = self.patchData(pixel)
            # current_Dp = 1
            
            current_Pp = current_Cp * (current_Dp ** self.data_significance) # Pp to change into matrix
            
            if current_Pp >= Pp :
                Pp = current_Pp
                Cp = current_Cp
                max_pixel = pixel

        # print("max_pixel", max_pixel)
        # print("Pp, Cp", Pp, Cp)

        return max_pixel, Cp


    # Get the optimal patch to use for the filling

    def distance(self, target_patch, candidate_patch, mask_patch) :

        mask_patch = np.expand_dims(mask_patch, axis=2)
        
        return np.sum(((target_patch - candidate_patch) * mask_patch) ** 2) / np.sum(mask_patch)

    def getSearchBoundaries(self, target_pixel) :
            
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

    def getOptimalPatch(self, target_pixel) :
        
        n, m = target_pixel

        upper_i, lower_i, upper_j, lower_j = self.getSearchBoundaries(target_pixel)
        
        
        target_patch = self.getPatch(self.image, target_pixel)         
        mask_patch = self.getPatch(self.mask, target_pixel)
        
        threshold_targets = []

        for i in range(lower_i, upper_i + 1) :
            for j in range(lower_j, upper_j + 1) :
                # if candidate patch is in part in mask then skip            
                if np.any(self.getPatch(self.mask, (i, j)) == 0) :
                    continue
                
                # compute the distance between the center pixels and only add to the array
                # if the distance is less than the center similarity threshold 
                d_center = np.sqrt(np.sum((self.image[n, m, :] - self.image[i, j, :]) ** 2))
                if self.threshold == None or d_center < self.threshold :
                    threshold_targets.append((i, j, d_center))
                    
        threshold_targets.sort(key = lambda x : x[2])
        threshold_targets = threshold_targets[:int(math.sqrt(len(threshold_targets)))]

        optimal_patch = (0, 0)
        optimal_distance = 1e9

        for i, j, d_center in threshold_targets :
            candidate_patch = self.getPatch(self.image, (i, j))
            current_distance = self.distance(target_patch, candidate_patch, mask_patch)
            
            if current_distance < optimal_distance :
                optimal_patch = (i, j)
                optimal_distance = current_distance
        
        return optimal_patch

    def updateConfidence(self, Cp, target_pixel) :   
        
        i, j = target_pixel

        self.confidence[i - self.offset : i + self.offset + 1, j - self.offset : j + self.offset + 1] = \
            self.getPatch(self.confidence, target_pixel) + (1 - self.getPatch(self.mask, target_pixel)) * Cp

    def fillPatch(self, target_pixel, opt_patch) :
        
        n, m = target_pixel
        i, j = opt_patch
        
        un, dn, lm, rm = n - self.offset, n + self.offset + 1, m - self.offset, m + self.offset + 1
        ui, di, lj, rj = i - self.offset, i + self.offset + 1, j - self.offset, j + self.offset + 1
        
        mask_patch = self.mask[un: dn, lm: rm]
        mask_patch = np.expand_dims(mask_patch, axis=2)
        
        self.image[un: dn, lm: rm, :] = self.image[un: dn, lm: rm, :] * mask_patch + self.image[ui: di, lj: rj, :] * (1 - mask_patch)
        
        self.mask[un: dn, lm: rm] = 1

    def _updateBorderPixel(self, i, j) :

        if self.isBorderPixel(i, j) :
            self.border_pxls.add((i, j))
        
        else :
            self.border_pxls.discard((i, j))

    def _updateBorder(self, target_pixel) :

        n, m = target_pixel

        upper_i = min(self.mask.shape[0] - self.offset, n + self.offset + 1)
        lower_i = max(self.offset, n - self.offset)
        upper_j = min(self.mask.shape[1] - self.offset, m + self.offset + 1)
        lower_j = max(self.offset, m - self.offset)
        
        for i in range(lower_i - 1, upper_i - 1) :
            for j in range(lower_j - 1, upper_j - 1) :
                self.border_pxls.discard((i, j))

        for i in range(max(self.offset, lower_i - 1), lower_i + 1) :
            for j in range(max(lower_j - 1, self.offset), min(upper_j + 1, self.mask.shape[1] - self.offset)) :
                self._updateBorderPixel(i, j)

        for i in range(upper_i - 1, min(upper_i + 1, self.mask.shape[0] - self.offset)) :
            for j in range(max(lower_j - 1, self.offset), min(upper_j + 1, self.mask.shape[1] - self.offset)) :
                self._updateBorderPixel(i, j)

        for j in range(max(self.offset, lower_j - 1), lower_j + 1) :
            for i in range(max(lower_i - 1, self.offset), min(upper_i + 1, self.mask.shape[0] - self.offset)) :
                self._updateBorderPixel(i, j)

        for j in range(upper_j - 1, min(upper_j + 1, self.mask.shape[1] - self.offset)) :
            for i in range(max(lower_i - 1, self.offset), min(upper_i + 1, self.mask.shape[0] - self.offset)) :
                self._updateBorderPixel(i, j)

        



    # __init__ method 
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
        self.start_zeros = None

    # inpainting methods
    def _inpaintingIteration(self) :

        """
            Utility method for an executing an inpainting iteration, updating
            the variable values accordingly, which is used in the later methods
            to execute the inpainting algorithm
        """

        progress = 100

        target_pixel, Cp = self.getMaxPriority()

        opt_patch = self.getOptimalPatch(target_pixel) # Most time-consuming function

        self.updateConfidence(Cp, target_pixel)

        self.fillPatch(target_pixel, opt_patch)

        start = time.time()
        self.progress = (1 - np.sum((1 - self.mask)) / self.start_zeros) * 100

        self._updateBorder(target_pixel)

    
    def inpaint(self, image, mask) :
        """
            Main method to handle the inpainting
            Parameters:
                - image: the image to inpaint
                - mask: the mask of the image to inpaint, denoting the masked area by 0's
            and the rest of the image by 1's
        """
        self.image = image
        self.mask = mask
        self.confidence = np.copy(mask)
        self.normalize()
        
        self.start_zeros = np.sum((1 - mask))
        self.getBorderPx()

        while len(self.border_pxls) != 0 :
            start = time.time()
            
            self._inpaintingIteration()
            print(time.time() - start)
            
            print("Almost there ! ===> {:.1f}/{}".format(self.progress, 100), sep='\n')
            
        return self.image

    def inpaintWithSteps(self, image, mask) :

        """
            Main method to handle the inpainting but returning the steps at each iteration
            Parameters:
                - image: the image to inpaint
                - mask: the mask of the image to inpaint, denoting the masked area by 0's
            and the rest of the image by 1's
        """

        self.image = image
        self.mask = mask
        self.confidence = np.copy(self.mask)
        self.normalize()
        
        self.start_zeros = np.sum((1 - self.mask))
        self.getBorderPx()

        while len(self.border_pxls) != 0 :
            
            start = time.time()

            self._inpaintingIteration()
            print(time.time() - start)
            
            print("Almost there ! ===> {:.1f}/{}".format(self.progress, 100), sep='\n')
            
            yield self.image, self.mask, self.progress