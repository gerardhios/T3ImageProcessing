from pathlib import Path
import cv2 as cv
import numpy as np

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@TODO: Implementar el F4                                                             @
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def binarizar(img, w_opt, b_opt):
    newimg = np.zeros(img.shape, dtype=np.uint8)
    for row, col in np.ndindex(img.shape):
        newimg[row, col] = b_opt if img[row, col] < 125 else w_opt
    return newimg

def dilatation(img, k_size):
    newimg = np.zeros(img.shape, dtype=np.uint8)
    for row, col in np.ndindex(img.shape):
        neighboors = neighborhoodCoordinates(row, col, k_size, img.shape)
        for coord in neighboors:
            if img[coord[0], coord[1]] == 1:
                newimg[row, col] = 1
                break
    return newimg

def erosion(img, k_size):
    newimg = np.zeros(img.shape, dtype=np.uint8)
    for row, col in np.ndindex(img.shape):
        neighbors = neighborhoodCoordinates(row, col, k_size, img.shape)
        sum = 0
        for coord in neighbors:
            sum += img[coord[0], coord[1]]
        if sum == k_size:
            newimg[row, col] = 1
    return newimg

# Check if the pixel is a border pixel
def isBorderPixel(row, col, img, search_size):
    neighboors = neighborhoodCoordinates(row, col, search_size, img.shape)
    for coord in neighboors:
        if img[coord[0], coord[1]] == 0:
            return True
    return False

# list of coordinates of the neigborhood
def neighborhoodCoordinates(row, col, size, img_shape):
    # Create a list to store the coordinates
    coordinates = []
    ret = []
    if size == 4:
        coordinates = [(row, col+1, 0), (row+1, col, 1), (row, col-1, 2), (row-1, col, 3)]
    elif size == 8:
        coordinates = [(row, col+1, 0), (row+1, col+1, 1), (row+1, col, 2), (row+1, col-1, 3) \
                       ,(row, col-1, 4), (row-1, col-1, 5), (row-1, col, 6), (row-1, col+1, 7)]
    # Filter the coordinates that are out of the image
    for coord in coordinates:
        if coord[0] < 0 or coord[1] < 0 or coord[0] >= img_shape[0] or coord[1] >= img_shape[1]:
            continue
        else:
            ret.append(coord)
    return ret

def freemanChainCodeAlgorithm(row, col, img, size, p_pixel=None):
    # Get the coordinates of the neighborhood
    neighboors = neighborhoodCoordinates(row, col, size, img.shape)
    # Iterate over the neighborhood
    for coord in neighboors:
        # If the pixel is black
        if img[coord[0], coord[1]] == 1:
            if size == 4:
                # If the pixel is a border pixel
                if isBorderPixel(coord[0], coord[1], img, 8):
                    # If the pixel is not the previous pixel
                    if (coord[0], coord[1]) != p_pixel:
                        # Return the chain code and the next pixel coordinates
                        return coord[2], (coord[0], coord[1])
            elif size == 8:
                # If the pixel is a border pixel
                if isBorderPixel(coord[0], coord[1], img, 4):
                    # If the pixel is not in the previous pixels
                    if (coord[0], coord[1]) not in p_pixel:
                        # Return the chain code and the next pixel coordinates
                        return coord[2], (coord[0], coord[1])


# Function to apply the freeman 4 chain code algorithm to a binary image
def freeman4ChainCode(img):
    pass
    return 

# Function to apply the freeman 4 chain code algorithm to a binary image
def freeman8ChainCode(img):
    chain = []
    p_pixels = []
    n_pixel = f_pixel = None
    # Iterate over the image find the first black pixel
    for row, col in np.ndindex(img.shape):
        # If the pixel is black
        if img[row, col] == 1:
            # If the pixel is a border pixel
            if isBorderPixel(row, col, img, 4):
                # Save the first pixel
                f_pixel = (row, col)
                # Apply the freeman 8 chain code algorithm
                ret = freemanChainCodeAlgorithm(row, col, img, 8, p_pixels)
                chain.append(ret[0])
                # p_pixels.append(f_pixel)
                n_pixel = ret[1]
                break
    while True:
        ret = freemanChainCodeAlgorithm(n_pixel[0], n_pixel[1], img, 8, p_pixels)
        chain.append(ret[0])
        p_pixels.append(n_pixel)
        n_pixel = ret[1]
        if n_pixel == (195, 147):
            print("")
        if n_pixel == f_pixel:
            break
    return chain

if __name__ == "__main__":
    # Read the image
    for i in range(1, 16):
        if i < 10:
            imgpath = Path(f"img/0{i}.png")
        else:
            imgpath = Path(f"img/{i}.png")
        img = cv.imread(str(imgpath), cv.IMREAD_GRAYSCALE)
        cv.imshow("Original", img)
        # Binarize the image
        img_b = binarizar(img, 255, 0)
        cv.imshow("Binarized image", img_b)
        # Save the binarized image
        # cv.imwrite("img/13_binarized.png", img_b)
        img_b = binarizar(img, 0, 1)
        # Apply the 3x3 dilatation
        img_d = dilatation(img_b, 8)
        img_d = dilatation(img_d, 8)
        img_d = dilatation(img_d, 8)
        # Apply the 3x3 erosion
        img_e = erosion(img_d, 8)
        # Obtain the border
        img_og = img_e.copy()
        img_e = erosion(img_e, 8)
        img_e = erosion(img_e, 8)
        img_e = erosion(img_e, 8)
        img_f = img_og - img_e
        img_f_v = img_f.copy()
        for row, col in np.ndindex(img_f.shape):
            if img_f_v[row, col] == 0:
                img_f_v[row, col] = 255
        cv.imshow("Border", img_f_v)
        # Save the border image
        # cv.imwrite("img/13_border.png", img_f_v)
        # Apply the freeman chain code algorithm
        # chain = freeman4ChainCode(img)
        # for i in chain:
        #     print(i, end="")
        chain = freeman8ChainCode(img_f)
        for i in chain:
            print(i, end="")
        cv.waitKey(0)
        cv.destroyAllWindows()