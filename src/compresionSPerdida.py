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

def freemanChainCodeAlgorithm(row, col, img, size, p_pixel):
    # Get the coordinates of the neighborhood
    neighboors = neighborhoodCoordinates(row, col, size, img.shape)
    # Iterate over the neighborhood
    for coord in neighboors:
        # If the pixel is black
        if img[coord[0], coord[1]] == 0:
            # If the pixel is not the previous pixel
            if (coord[0], coord[1]) != p_pixel:
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
    # Iterate over the image
    for row, col in np.ndindex(img.shape):
        # If the pixel is black
        if img[row, col] == 0:
            # If the pixel is a border pixel
            if isBorderPixel(row, col, img, 4):
                # Apply the freeman 4 chain code algorithm
                chain.append(freemanChainCodeAlgorithm(row, col, img, 8))
    return chain

img = np.zeros((7, 7), dtype=np.uint8)
for row, col in np.ndindex(img.shape):
    img[row, col] = 255
img[1, 1] = 0
img[1, 2] = 0
img[1, 3] = 0
img[1, 4] = 0
img[1, 5] = 0
img[2, 1] = 0
img[2, 5] = 0
img[3, 1] = 0
img[3, 5] = 0
img[4, 1] = 0
img[4, 5] = 0
img[5, 1] = 0
img[5, 2] = 0
img[5, 3] = 0
img[5, 4] = 0
img[5, 5] = 0
print(img)
code_chain = freeman4ChainCode(img)
for i in code_chain:
    print(i, end=" ")