# class Inpainter
The Inpainter object that handles the inpainting
    
### _normalize(self)

Utility method to normalize the image before running
the inpainting algorithm

<details>
<summary>Click to see code</summary>
    
{% highlight python %}

def _normalize(self) :
    self.image = self.image/np.max(self.image)
{% endhighlight %}

</details>
<br>

### _getPatch(self, array, center)

Utility method to get patch of image or mask or any other numpy array
Note the array must is supposed to be a mask or an image so it should
be 2D or 3D (RGB/RGBA) format

**Parameters :**

    array : array to get the patch from

    center : center of the patch to extract

<details>
<summary>Click to see code</summary>
  

{% highlight python %}

def _getPatch(self, array, center) :
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
 
{% endhighlight %}
  
</details>
<br>

### _isBorderPixel(self, n, m)

Utility method that takes a pixel position (n, m) and a mask as input and returns
whether the pixel in that position is a border pixel or not

**Parameters :**

    n: coordinate of pixel on the first axis
    
    m: coordinate of pixel on the second axis 

<details>
<summary>Click to see code</summary>
  

{% highlight python %}
def _isBorderPixel(self, n, m) :

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
{% endhighlight %}
  
</details>
<br>

### _getBorderPx(self)

Utility method to get all the border pixels in the image
excluding of course those who have a patch that partially goes 
over the border

<details>
<summary>Click to see code</summary>
 
{% highlight python %}
def _getBorderPx(self) :
    self.border_pxls = set()

    upper_i = self.mask.shape[0] - self.offset - 1
    lower_i = self.offset
    upper_j = self.mask.shape[1] - self.offset - 1
    lower_j = self.offset

    for i in range(lower_i, upper_i + 1) :
        for j in range(lower_j, upper_j + 1) :
            if self._isBorderPixel(i, j) :
                self.border_pxls.add((i, j))
{% endhighlight %}
  
</details>
<br>

## Methods to calculate the maximum priority

### _patchConfidence(self, center) :

Utility method to calculate the confidence of a chosen patch

**Parameters :**

    center: tuple of the coordinated of the center of patch 
    
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _patchConfidence(self, center) :
    return np.sum(self._getPatch(self.confidence, center)) / self.patch_size**2
{% endhighlight %}
  
</details>   
<br>    

## _calcNormalMatrix(self):

Utility method to precalculate the normal vector to each pixel in the mask
at each iteration

<details>
<summary>Click to see code</summary>

{% highlight python %}
def _calcNormalMatrix(self):
    x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
    y_kernel = np.array([[.25, .5, .25], [0, 0, 0], [-.25, -.5, -.25]])

    x_normal = convolve(self.mask.astype(float), x_kernel)
    y_normal = convolve(self.mask.astype(float), y_kernel)

    normal = np.dstack((x_normal, y_normal))

    norm = np.sqrt(normal[:, :, 0]**2 + normal[:, :, 1]**2)
    norm[norm == 0] = 1

    unit_normal = -normal / np.expand_dims(norm, axis=2)
    self.normal_mask = unit_normal
{% endhighlight %}
  
</details>
<br>

### _getNormalPatch(self, center):
Utility method to calculate the normal vector to a chosen patch

**Parameters :**

    center: tuple of the coordinates of the center of patch

<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _getNormalPatch(self, center):
    i, j = center
    return self.normal_mask[i, j]
{% endhighlight %}
  
</details>
<br>

## Methods to get the normal and gradient vectors to a patch

### _calcGradientMask(self):
    
Utility method to calculate the gradient vector to each pixel in the mask
<details>
<summary>Click to see code</summary>
  
{% highlight python %}  
def _calcGradientMask(self):        
    grey_image = rgb2gray(self.image)
    grey_image[self.mask == 0] = None

    self.gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
    self.gradient_val = np.sqrt(self.gradient[0]**2 + self.gradient[1]**2)
{% endhighlight %}
  
</details>
<br>

### _getGradientPatch(self, center) :

Utility method to get the maximum norm of a gradient vector 
to the pixels of a chosen patch

**Parameters :**

    center: tuple of the coordinates of the center of patch
    
<details>
<summary>Click to see code</summary>
 
{% highlight python %}
def _getGradientPatch(self, center) :
    max_gradient = np.zeros([2])

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
{% endhighlight %}
  
</details>
<br>

### _prepareDataUtils(self) :

Utility method to prepare the normal and gradient matrices for all
the image pixels using matrix multiplications to optimize the runtime,
instead of doing the computation for each pixel

<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _prepareDataUtils(self) :
    self._calcNormalMatrix()
    self._calcGradientMask()
{% endhighlight %}
  
</details>
<br>

### _patchData(self, center) :

Method that calculates the data term for a given patch, according to the 
formula in the paper

**Parameters :**

    center: tuple of the coordinates of the center of patch
                
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _patchData(self, center) :
    normal_vector = self._getNormalPatch(center)
    max_gradient = self._getGradientPatch(center)

    data = np.abs(np.dot(normal_vector, max_gradient.T)) / self.alpha

    return data
{% endhighlight %}

</details>
<br>

## Actually calculate the maximum priority 

### _getMaxPriority(self) :

Utility method that finds the max priority border pixel to fill its
respective patch 
            
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _getMaxPriority(self) :
    Pp, Cp = 0, 0
    max_pixel = (0, 0)
    # print("border", len(self.border_pxls))
    self._prepareDataUtils()
    
    # loop over the border pixels to get the maximum priority one
    for pixel in self.border_pxls :

        n, m = pixel

        # if n - offset < 0 or n + offset + 1 > image.shape[0] or m - offset < 0 or m + offset + 1 > image.shape[1] :
        #     continue

        current_Cp = self._patchConfidence(pixel)
        current_Dp = self._patchData(pixel)
        # current_Dp = 1

        # Compute the priority according to the formula in the paper
        current_Pp = current_Cp * (current_Dp ** self.data_significance)

        if current_Pp >= Pp :
            Pp = current_Pp
            Cp = current_Cp
            max_pixel = pixel

    return max_pixel, Cp
{% endhighlight %}
  
</details>
<br>

## Get the optimal patch to use for the filling

### _distance(self, target_patch, candidate_patch, mask_patch) :

Utility method that calculates the distance between the candidate patch and 
the target patch

**Parameters :**

    target_patch: tuple of coordinates of target patch 
    
    candidate_patch: tuple of coordinates of candidate patch 
    
    mask_patch: tuple of coordinates of mask patch 
                
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _distance(self, target_patch, candidate_patch, mask_patch) :
    mask_patch = np.expand_dims(mask_patch, axis=2)

    return np.sum(((target_patch - candidate_patch) * mask_patch) ** 2) / np.sum(mask_patch)
{% endhighlight %}
  
</details>
<br>


### _getSearchBoundaries(self, target_pixel) :

Utility method to get the limits for the optimal patch search, according
to the local_radius parameter 

**Parameters :**

    target_pixel: tuple of the coordinates of the target pixel, center of the patch
    to fill
            
<details>
<summary>Click to see code</summary>
  
{% highlight python %}  
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
{% endhighlight %}
  
</details>
<br>

### _getOptimalPatch(self, target_pixel) :

Utility method for getting the optimal patch from the potential patches,
these are the ones that are fully included in the image (no pixel in masked
region), and are within a radius of the target pixel (specified by the parameter
local_radius in the __init__ method).

**Parameters :**

    target_pixel: tuple of the coordinates of the target pixel, center of the patch
    to fill
                
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _getOptimalPatch(self, target_pixel) :
    n, m = target_pixel

    upper_i, lower_i, upper_j, lower_j = self._getSearchBoundaries(target_pixel)

    # initialize variables
    target_patch = self._getPatch(self.image, target_pixel)         
    mask_patch = self._getPatch(self.mask, target_pixel)

    threshold_targets = []

    # loop to get the patches that verify the center similarity threshold
    for i in range(lower_i, upper_i + 1) :
        for j in range(lower_j, upper_j + 1) :
            # if candidate patch is in part in mask then skip            
            if np.any(self._getPatch(self.mask, (i, j)) == 0) :
                continue

            # compute the distance between the center pixels and only add to the array
            # if the distance is less than the center similarity threshold 
            d_center = np.sqrt(np.sum((self.image[n, m, :] - self.image[i, j, :]) ** 2))
            if self.threshold == None or d_center < self.threshold :
                threshold_targets.append((i, j, d_center))

    # Only take a subset of the closest thresholded target to 
    # speed up the algorithm
    threshold_targets.sort(key = lambda x : x[2])
    threshold_targets = threshold_targets[:int(math.sqrt(len(threshold_targets)))]

    # go through the filtered patches and then compute their distance 
    # to the target patch, to find the optimal one
    optimal_patch = (0, 0)
    optimal_distance = 1e9

    for i, j, d_center in threshold_targets :
        candidate_patch = self._getPatch(self.image, (i, j))
        current_distance = self._distance(target_patch, candidate_patch, mask_patch)

        if current_distance < optimal_distance :
            optimal_patch = (i, j)
            optimal_distance = current_distance

    return optimal_patch
{% endhighlight %}
  
</details>
<br>

### _updateConfidence(self, Cp, target_pixel) :
    
Utility method for updating the confidence matrix values in the target patch
according to the formula in the paper
            
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _updateConfidence(self, Cp, target_pixel) :
    i, j = target_pixel

    self.confidence[i - self.offset : i + self.offset + 1, j - self.offset : j + self.offset + 1] = \
        self._getPatch(self.confidence, target_pixel) + (1 - self._getPatch(self.mask, target_pixel)) * Cp
{% endhighlight %}
  
</details>
<br>

### _fillPatch(self, target_pixel, opt_patch) :

Utility method for filling the target patch with the optimal patch found 
in the previous steps in the masked region 
            
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _fillPatch(self, target_pixel, opt_patch) :
    n, m = target_pixel
    i, j = opt_patch

    un, dn, lm, rm = n - self.offset, n + self.offset + 1, m - self.offset, m + self.offset + 1
    ui, di, lj, rj = i - self.offset, i + self.offset + 1, j - self.offset, j + self.offset + 1

    mask_patch = self.mask[un: dn, lm: rm]
    mask_patch = np.expand_dims(mask_patch, axis=2)

    self.image[un: dn, lm: rm, :] = self.image[un: dn, lm: rm, :] * mask_patch + self.image[ui: di, lj: rj, :] * (1 - mask_patch)

    self.mask[un: dn, lm: rm] = 1
{% endhighlight %}
  
</details>
<br>

### _updateBorderPixel(self, i, j) :

Utility method for the main logic of updating pixels on the edge of the filled patch

**Parameters :**

    i, j: coordinates of the pixel
                
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _updateBorderPixel(self, i, j) :
    if self._isBorderPixel(i, j) :
        self.border_pxls.add((i, j))

    else :
        self.border_pxls.discard((i, j))
{% endhighlight %}
  
</details>
<br>

### _updateBorder(self, target_pixel) :
    
Utility method for an updating the set of border pixels after an iteration,
removing the pixels that are no longer on the border (inside the filled patch),
and adding the new ones in the set

**Parameters :**

    target_pixel: the center of the target patch filled by the algorithm 
                
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
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
{% endhighlight %}
  
</details>
<br>
        

### __init__(self, patch_size, local_radius, data_significance = 0, alpha = 1, threshold = None) :
    
Initiantes the inpainter object with parameters for the inpainting

**Parameters :**

    patch_size: the patch size the algorithm uses to fill the mask at each iteration
    
    local_radius: specify a radius to limit the search for the optimal to a 
        neighboring region 
        
    data_significance: the significance accorded to the data term, 0 meaning 
        totally ignored and 1 meaning full significance
        
    alpha: the alpha term in the formula of the data term
    
    threshold: the center similarity threshold to use in order to reduce the complexity
        by leaving out patches with central pixels that are too different (difference bigger
        than the threshold) 
                    
<details>
<summary>Click to see code</summary>
 
{% highlight python %}
def __init__(self, patch_size, local_radius, data_significance = 0, alpha = 1, threshold = None) :
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
{% endhighlight %}
  
</details>
<br>

## inpainting methods

### _inpaintingIteration(self) :
    
Utility method for an executing an inpainting iteration, updating
the variable values accordingly, which is used in the later methods
to execute the inpainting algorithm

<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def _inpaintingIteration(self) :

    progress = 100

    target_pixel, Cp = self._getMaxPriority()

    opt_patch = self._getOptimalPatch(target_pixel) # Most time-consuming function

    self._updateConfidence(Cp, target_pixel)

    self._fillPatch(target_pixel, opt_patch)

    start = time.time()
    self.progress = (1 - np.sum((1 - self.mask)) / self.start_zeros) * 100

    self._updateBorder(target_pixel)
{% endhighlight %}
  
</details>
<br>
    
### inpaint(self, image, mask) :

Main method to handle the inpainting

**Parameters :**

    - image: the image to inpaint
    - mask: the mask of the image to inpaint, denoting the masked area by 0's
and the rest of the image by 1's

        
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def inpaint(self, image, mask) :
    self.image = image
    self.mask = mask
    self.confidence = np.copy(mask)
    self._normalize()

    self.start_zeros = np.sum((1 - mask))
    self._getBorderPx()

    while len(self.border_pxls) != 0 :
        start = time.time()

        self._inpaintingIteration()
        print(time.time() - start)

        print("Almost there ! ===> {:.1f}/{}".format(self.progress, 100), sep='\n')

    return self.image
{% endhighlight %}
  
</details>
<br>

### inpaintWithSteps(self, image, mask) :

Main method to handle the inpainting but returning the steps at each iteration
**Parameters :**

    - image: the image to inpaint

    - mask: the mask of the image to inpaint, denoting the masked area by 0's
and the rest of the image by 1's
    
<details>
<summary>Click to see code</summary>
  
{% highlight python %}
def inpaintWithSteps(self, image, mask) :
    self.image = image
    self.mask = mask
    self.confidence = np.copy(self.mask)
    self._normalize()

    self.start_zeros = np.sum((1 - self.mask))
    self._getBorderPx()

    while len(self.border_pxls) != 0 :

        start = time.time()

        self._inpaintingIteration()
        print(time.time() - start)

        print("Almost there ! ===> {:.1f}/{}".format(self.progress, 100), sep='\n')

        yield self.image, self.mask, self.progress
{% endhighlight %}
  
</details>
<br>
