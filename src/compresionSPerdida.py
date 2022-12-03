from pathlib import Path
import cv2 as cv
import numpy as np

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@TODO: Implementar el recorrido a partir de un punto inicial                         @
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def binarizar(img):
    newimg = np.zeros(img.shape, dtype=np.uint8)
    for row, col in np.ndindex(img.shape):
        newimg[row, col] = 0 if img[row, col] < 110 else 255
    return newimg

# Check if the pixel is a border pixel
def isBorderPixel(row, col, img, search_size):
    neighboors = neighborhoodCoordinates(row, col, search_size, img.shape)
    for coord in neighboors:
        if img[coord[0], coord[1]] == 255:
            return True
    return False

# list of coordinates of the neigborhood
def neighborhoodCoordinates(row, col, size, img_shape):
    # Create a list to store the coordinates
    coordinates = []
    if size == 4:
        coordinates = [(row, col+1, 0), (row+1, col, 1), (row, col-1, 2), (row-1, col, 3)]
    elif size == 8:
        coordinates = [(row, col+1, 0), (row+1, col+1, 1), (row+1, col, 2), (row+1, col-1, 3) \
                       ,(row, col-1, 4), (row-1, col-1, 5), (row-1, col, 6), (row-1, col+1, 7)]
    # Filter the coordinates that are out of the image
    for coord in coordinates:
        if coord[0] < 0 or coord[1] < 0 or coord[0] >= img_shape[0] or coord[1] >= img_shape[1]:
            coordinates.remove(coord)
    return coordinates

def freemanChainCodeAlgorithm(row, col, img, size, p_pixel=None):
    # Get the coordinates of the neighborhood
    neighboors = neighborhoodCoordinates(row, col, size, img.shape)
    # Iterate over the neighborhood
    for coord in neighboors:
        # If the pixel is black
        if img[coord[0], coord[1]] == 0:
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
    chain = []
    p_pixel = n_pixel = f_pixel = None
    # Iterate over the image find the first black pixel
    for row, col in np.ndindex(img.shape):
        # If the pixel is black
        if img[row, col] == 0:
            # If the pixel is a border pixel
            if isBorderPixel(row, col, img, 8):
                # Save the first pixel
                f_pixel = (row, col)
                # Apply the freeman 4 chain code algorithm
                ret = freemanChainCodeAlgorithm(row, col, img, 4, p_pixel)
                chain.append(ret[0])
                p_pixel = f_pixel
                n_pixel = ret[1]
                break
    while True:
        ret = freemanChainCodeAlgorithm(n_pixel[0], n_pixel[1], img, 4, p_pixel)
        chain.append(ret[0])
        p_pixel = n_pixel
        n_pixel = ret[1]
        if n_pixel == f_pixel:
            break
    return chain

# Function to apply the freeman 4 chain code algorithm to a binary image
def freeman8ChainCode(img):
    chain = []
    p_pixels = []
    n_pixel = f_pixel = None
    # Iterate over the image find the first black pixel
    for row, col in np.ndindex(img.shape):
        # If the pixel is black
        if img[row, col] == 0:
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
        # if n_pixel == (178, 244):
        #     print("")
        if n_pixel == f_pixel:
            break
    return chain

if __name__ == "__main__":
    # Read the image
    imgpath = Path("img/15.png")
    img = cv.imread(str(imgpath), cv.IMREAD_GRAYSCALE)
    # Binarize the image
    img = binarizar(img)
    # Save the binarized image
    # cv.imwrite("img/05_binarized.png", img)
    # Apply the freeman chain code algorithm
    # chain = freeman4ChainCode(img)
    # for i in chain:
    #     print(i)
    chain = freeman8ChainCode(img)
    for i in chain:
        print(i, end="")