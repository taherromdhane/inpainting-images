import numpy as np
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray

def normalize(image) :
    return image/np.max(image)

def isBorderPixel(n, m, mask) :
    # function that takes a pixel position (n, m) and a mask as input and returns
    # wether the pixel in that position is a border pixel or not
    
    if mask[n][m] == 0 :
        return False
    
    for i in range(-1, 2) :
        for j in range(-1, 2) :
            if i == 0 and j == 0 or n+i<0 or m+j<0:
                continue
            try :
                if mask[n + i][m + j] == 0 :
                    return True
            except IndexError :
                continue
                
    return False

def getBorderPx(mask) :
    border_pxls = set()
    
    for i in range(mask.shape[0]) :
        for j in range(mask.shape[1]) :
            if isBorderPixel(i, j, mask) :
                border_pxls.add((i, j))
    
    return border_pxls

def patchConfidence(center, confidence, mask, patch_size) :
    
    i, j = center
    offset = patch_size//2
    
    return np.sum(confidence[i - offset : i + offset + 1, j - offset : j + offset + 1]) / patch_size**2
    
def calcNormalMatrix(center, mask):
    
    x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
    y_kernel = np.array([[.25, .5, .25], [0, 0, 0], [-.25, -.5, -.25]])
    
    x_normal = convolve(mask.astype(float), x_kernel)
    y_normal = convolve(mask.astype(float), y_kernel)
    
    normal = np.dstack((x_normal, y_normal))[1, 1, :]
    
    norm = np.sqrt(normal[0]**2 + normal[1]**2)
    if norm == 0 :
        norm = 1

    unit_normal = -normal/norm
    return unit_normal
    
def calcGradientMatrix(center, image, mask, patch_size):
    
    # TODO: find a better method to calc the gradient
    height, width = image.shape[:2]
    
    
    grey_image = rgb2gray(image)
    grey_image[mask == 1] = None

    gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
    gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
    max_gradient = np.zeros([2])
    
    offset = patch_size//2
    i, j = center
    
    patch_y_gradient = gradient[0][i - offset : i + offset + 1, j - offset : j + offset + 1]
    patch_x_gradient = gradient[1][i - offset : i + offset + 1, j - offset : j + offset + 1]
    patch_gradient_val = gradient_val[i - offset : i + offset + 1, j - offset : j + offset + 1]

    patch_max_pos = np.unravel_index(
        patch_gradient_val.argmax(),
        patch_gradient_val.shape
    )

    max_gradient[0] = patch_y_gradient[patch_max_pos]
    max_gradient[1] = patch_x_gradient[patch_max_pos]

    return max_gradient
    
    
def patchData(center, image, mask, alpha, patch_size) :
    
    offset = 1
    i, j = center
    patch_mask = mask[i - offset : i + offset + 1, j - offset : j + offset + 1]
    normal = calcNormalMatrix(center, patch_mask)
    max_gradient = calcGradientMatrix(center, image, mask, patch_size)
    
    data = np.abs(np.dot(normal, max_gradient.T)) / alpha
    
    return data
   

def getMaxPriority(border_pxls, confidence, image, mask, alpha, patch_size) :
    
    Pp, Cp = 0, 0
    max_pixel = (0, 0)
    
    for pixel in border_pxls :
        
        n, m = pixel
        offset = patch_size//2
        
        if n - offset < 0 or n + offset + 1 > image.shape[0] or m - offset < 0 or m + offset + 1 > image.shape[1] :
            continue
        
        current_Cp = patchConfidence(pixel, confidence, mask, patch_size)
        # current_Dp = patchData(pixel, image, mask, alpha, patch_size)
        current_Dp = 1
        
        current_Pp = current_Cp * current_Dp # Pp to change into matrix
        
        if current_Pp > Pp :
            Pp = current_Pp
            Cp = current_Cp
            max_pixel = pixel
            
    return pixel, Cp

def distance(target_patch, candidate_patch, mask_patch) :
     
#     print((target_patch - candidate_patch) * mask_patch)  
#     print(((target_patch - candidate_patch) * mask_patch) ** 2)
    
    mask_patch = np.expand_dims(mask_patch, axis=2)

    return np.sum(((target_patch - candidate_patch) * mask_patch) ** 2) / np.sum(mask_patch)

def getOptimalPatch(image, mask, target_patch, patch_size, local_radius = None) :
    
    n, m = target_patch
    
    offset = patch_size//2
    
    if local_radius :
        upper_i = min(n + local_radius, image.shape[0] - offset - 1)
        lower_i = max(n - local_radius, offset)
        upper_j = min(m + local_radius, image.shape[1] - offset - 1)
        lower_j = max(m - local_radius, offset)
    else :
        upper_i = image.shape[0] - offset - 1
        lower_i = offset
        upper_j = image.shape[1] - offset - 1
        lower_j = offset
#     print("i : {} - {}, j : {} - {}".format(lower_i, upper_i, lower_j, upper_j))
    
    optimal_patch = (0, 0)
    optimal_distance = 1e9
    
    for i in range(lower_i, upper_i + 1) :
        for j in range(lower_j, upper_j + 1) :
            
#             print(i, "-", j)
            
            if np.any(mask[i - offset : i + offset + 1, j - offset : j + offset + 1] == 0) :
                continue
            
            target_patch = image[n - offset : n + offset + 1, m - offset : m + offset + 1, :]
            
            # if candidate patch in part in mask break
            candidate_patch = image[i - offset : i + offset + 1, j - offset : j + offset + 1, :]            
            mask_patch = mask[n - offset : n + offset + 1, m - offset : m + offset + 1]
            
#             print("target patch : \n", target_patch)            
#             print("candidate patch : \n\n", candidate_patch)            
#             print("mask patch : \n", mask_patch)
            
            current_distance = distance(target_patch, candidate_patch, mask_patch)
            
#             print("current distance : {}".format(current_distance))
            
            if current_distance < optimal_distance :
                optimal_patch = (i, j)
                optimal_distance = current_distance
    
    return optimal_patch

def updateConfidence(confidence, Cp, target_patch, mask, patch_size) :   
    
    i, j = target_patch
    offset = patch_size//2

    confidence[i - offset : i + offset + 1, j - offset : j + offset + 1] = confidence[i - offset : i + offset + 1, j - offset : j + offset + 1] + (1 - mask[i - offset : i + offset + 1, j - offset : j + offset + 1]) * Cp
    
    return confidence

def fillPatch(image, mask, target_patch, opt_patch, patch_size) :
    
    n, m = target_patch
    i, j = opt_patch
    offset = patch_size//2
    
    un, dn, lm, rm = n - offset, n + offset + 1, m - offset, m + offset + 1
    ui, di, lj, rj = i - offset, i + offset + 1, j - offset, j + offset + 1
    
#     mask = np.expand_dims(mask, axis=2)
#     print("image * mask \n", image * mask)
    
    mask_patch = mask[un: dn, lm: rm]
    mask_patch = np.expand_dims(mask_patch, axis=2)
    
#     print("image[un: dn, lm: rm, :] * mask_patch \n", image[un: dn, lm: rm, :] * mask_patch)
#     print("image[ui: di, lj: rj, :] * (1 - mask_patch) \n", image[ui: di, lj: rj, :] * (1 - mask_patch))
    
    image[un: dn, lm: rm, :] = image[un: dn, lm: rm, :] * mask_patch + image[ui: di, lj: rj, :] * (1 - mask_patch)
    
    mask[un: dn, lm: rm] = 1
    
    return image, mask

def inpaint(image, mask, patch_size=9, alpha=1, local_radius=500) :
    
    # assert patch_size is an odd number
    assert(patch_size%2 == 1)
    
    confidence = np.copy(mask)
    image = normalize(image) # * np.expand_dims(mask, axis=2)
    
    start_zeros = np.sum((1 - mask))

    # change to identify border, then calculate priorities
    while True :

        border_pxls = getBorderPx(mask)
        if len(border_pxls) == 0 :
            break
            
        target_patch, Cp = getMaxPriority(border_pxls, confidence, image, mask, alpha, patch_size)

        opt_patch = getOptimalPatch(image, mask, target_patch, patch_size, local_radius)

        confidence = updateConfidence(confidence, Cp, target_patch, mask, patch_size)

        image, mask = fillPatch(image, mask, target_patch, opt_patch, patch_size)
        
        print("Almost there ! ===> {:.1f}/{}".format((1 - np.sum((1 - mask)) / start_zeros) * 100, 100), sep='\n')
        
    return image
