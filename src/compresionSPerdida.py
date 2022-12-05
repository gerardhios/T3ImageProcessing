from pathlib import Path
import cv2 as cv
import numpy as np
import os

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@TODO: Cambiar las direcciones del F4 y probarlo con una imagen                      @
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
            if size == 8:
                # If the pixel is a border pixel
                if isBorderPixel(coord[0], coord[1], img, 4):
                    # If the pixel is not in the previous pixels
                    if (coord[0], coord[1]) not in p_pixel:
                        # Return the chain code and the next pixel coordinates
                        return coord[2], (coord[0], coord[1])



# Function to apply the freeman 8 chain code algorithm to a binary image
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
        # if n_pixel == (195, 147):
        #     print("")
        if n_pixel == f_pixel:
            break
    return chain

# Function to apply the freeman 4 chain code algorithm to a binary image
def freeman4ChainCode(img):
    aux = []
    chain = []
    y = x = None
    shape = (img.shape[0], img.shape[1])
    # Iterate over the image find the first black pixel
    for row, col in np.ndindex(shape):
        # If the pixel is black
        if img[row, col][0] == 1:
            # Save the first pixel
            # print((row, col))
            y = row
            x = col
            break
    # Mark the first pixel
    img[y, x][0] = 0

    if img[y][x - 1][1] != 1:
        aux.append(3)
        # Figure 1
        if img[y - 1][x - 1][1] != 1:
            if img[y - 1][x][1] != 1:
                # Right
                chain.append(0)
                if img[y - 1][x + 1][1] != 1:
                    if img[y][x + 1][1] != 1:
                        # Down
                        chain.append(1)
    while True:
        # if len(chain) > 3:
        #     print("Last chain codes:", chain[-3:])
        # if y == 256 and x == 188:
        #     print("")
        if img[y][x+1][0] == 1:
            img[y][x+1][0] = 0
            x += 1
            # Figure 2
            if img[y - 1][x][1] != 1:
                # Right
                chain.append(0)
                if img[y - 1][x + 1][1] != 1:
                    if img[y][x + 1][1] != 1:
                        # Down
                        chain.append(1)
                        if img[y + 1][x + 1][1] != 1:
                            if img[y + 1][x][1] != 1:
                                # Left
                                chain.append(2)

        elif img[y+1][x][0] == 1:
            img[y+1][x][0] = 0
            y += 1
            # Figure 3
            if img[y][x + 1][1] != 1:
                # Down
                chain.append(1)
                if img[y + 1][x + 1][1] != 1:
                    if img[y + 1][x][1] != 1:
                        # Left
                        chain.append(2)
                        if img[y + 1][x - 1][1] != 1:
                            if img[y][x-1][1] != 1:
                                # Up
                                chain.append(3)
        
        elif img[y][x-1][0] == 1:
            img[y][x-1][0] = 0
            x -= 1
            # Figure 4
            if img[y + 1][x][1] != 1:
                # Left
                chain.append(2)
                if img[y + 1][x - 1][1] != 1:
                    if img[y][x-1][1] != 1:
                        # Up
                        chain.append(3)
                        if img[y - 1][x - 1][1] != 1:
                            if img[y-1][x][1] != 1:
                                chain.append(0)
        
        elif img[y-1][x][0] == 1:
            img[y-1][x][0] = 0
            y -= 1
            # Figure 5
            if img[y][x - 1][1] != 1:
                # Up
                chain.append(3)
                if img[y - 1][x - 1][1] != 1:
                    if img[y - 1][x][1] != 1:
                        # Right
                        chain.append(0)
                        if img[y - 1][x + 1][1] != 1:
                            if img[y][x+1][1] != 1:
                                # Down
                                chain.append(1)
        
        elif img[y-1][x+1][0] == 1:
            img[y-1][x+1][0] = 0
            y -= 1
            x += 1
            # Figure 6
            if img[y][x - 1][1] != 1:
                # Up
                chain.append(3)
                if img[y - 1][x - 1][1] != 1:
                    if img[y - 1][x][1] != 1:
                        # Right
                        chain.append(0)
                        if img[y - 1][x + 1][1] != 1:
                            if img[y][x + 1][1] != 1:
                                # Down
                                chain.append(1)
                                if img[y + 1][x + 1][1] != 1:
                                    if img[y + 1][x][1] != 1:
                                        # Left
                                        chain.append(2)

        elif img[y+1][x+1][0] == 1:
            img[y+1][x+1][0] = 0
            y += 1
            x += 1
            # Figure 7
            if img[y - 1][x][1] != 1:
                # Right
                chain.append(0)
                if img[y - 1][x + 1][1] != 1:
                    if img[y][x + 1][1] != 1:
                        # Down
                        chain.append(1)
                        if img[y + 1][x + 1][1] != 1:
                            if img[y + 1][x][1] != 1:
                                # Left
                                chain.append(2)
                                if img[y + 1][x - 1][1] != 1:
                                    if img[y][x-1][1] != 1:
                                        chain.append(3)

        elif img[y+1][x-1][0] == 1:
            img[y+1][x-1][0] = 0
            y += 1
            x -= 1
            # Figure 8
            if img[y][x + 1][1] != 1:
                # Down
                chain.append(1)
                if img[y + 1][x + 1][1] != 1:
                    if img[y + 1][x][1] != 1:
                        # Left
                        chain.append(2)
                        if img[y + 1][x - 1][1] != 1:
                            if img[y][x-1][1] != 1:
                                # Up
                                chain.append(3)
                                if img[y - 1][x - 1][1] != 1:
                                    if img[y-1][x][1] != 1:
                                        # Right
                                        chain.append(0)

        elif img[y-1][x-1][0] == 1:
            img[y-1][x-1][0] = 0
            y -= 1
            x -= 1
            # Figure 9
            if img[y + 1][x][1] != 1:
                # Left
                chain.append(2)
                if img[y + 1][x - 1][1] != 1:
                    if img[y][x - 1][1] != 1:
                        # Up
                        chain.append(3)
                        if img[y - 1][x - 1][1] != 1:
                            if img[y - 1][x][1] != 1:
                                # Right
                                chain.append(0)
                                if img[y - 1][x + 1][1] != 1:
                                    if img[y][x+1][1] != 1:
                                        # Down
                                        chain.append(1)
        else:
            break
    while len(aux)>0:
        chain.append(aux.pop(0))
    return chain

def VCC(f4_chain):
    vcc = []
    convertion = [[1  ,2  ,"*",0  ],
                  [0  ,1  ,2  ,"*"],
                  ["*",0  ,1  ,2  ],
                  [2  ,"*",0  ,1  ]]
    
    for i in range(len(f4_chain) - 1):
        vcc.append(convertion[f4_chain[i]][f4_chain[i+1]])
    return vcc

def c3OT(f4_chain):
    c3ot = []
    aux = 1
    convertion = [[1  ,"*",2  ,"*"],
                  ["*",1  ,"*",2  ],
                  [2  ,"*",1  ,"*"],
                  ["*",2  ,"*",1  ]]
    
    for i in range(len(f4_chain) - 1):
        if f4_chain[i] == f4_chain[i+1]:
            c3ot.append(0)
        else:
            c3ot.append(convertion[aux][f4_chain[i+1]])
            aux = f4_chain[i]
    c3ot.append(1)
    return c3ot

def cAF8(f8_chain):
    af8 = []
    convertion = [[0,1,3,5,7,6,4,2],
                  [2,0,1,3,5,7,6,4],
                  [4,2,0,1,3,5,7,6],
                  [6,4,2,0,1,3,5,7],
                  [7,6,4,2,0,1,3,5],
                  [5,7,6,4,2,0,1,3],
                  [3,5,7,6,4,2,0,1],
                  [1,3,5,7,6,4,2,0]]

    for i in range(len(f8_chain) - 1):
        af8.append(convertion[f8_chain[i]][f8_chain[i+1]])
    af8.append(2)
    return af8

def read_image(index):
    if index < 10:
            imgpath = Path(f"img/0{index}.png")
    else:
        imgpath = Path(f"img/{index}.png")
    img = cv.imread(str(imgpath), cv.IMREAD_GRAYSCALE)
    cv.imshow("Original", img)
    return img, imgpath

def prepare_image(index):
    img, imgpath = read_image(index)
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
    img_e1 = erosion(img_e, 8)
    img_e = erosion(img_e1, 8)
    img_e = erosion(img_e, 8)
    img_f = img_og - img_e
    # Obtain a lighter border image
    img_f4 = img_og - img_e1
    img_f_v = img_f.copy()
    for row, col in np.ndindex(img_f.shape):
        if img_f_v[row, col] == 0:
            img_f_v[row, col] = 255
    cv.imshow("Border", img_f_v)
    # Save the border image
    # cv.imwrite(f"img/{index}_border.png", img_f_v)
    return img_f, img_f4, imgpath

def calculate_entropy(n_chars, chain):
    frecuency_lst = []
    for i in range(n_chars):
        frecuency_lst.append(chain.count(i))
    entropy = 0
    for i in range(n_chars):
        if frecuency_lst[i] != 0:
            entropy += (-(((frecuency_lst[i]) / len(chain)) * (np.log2(frecuency_lst[i] / len(chain))))) + entropy
    return entropy

def assign_code(nodes, label, result, prefix=''):
    childs = nodes[label]
    tree = {}
    if len(childs) == 2:
        tree['0'] = assign_code(nodes, childs[0], result, prefix + '0')
        tree['1'] = assign_code(nodes, childs[1], result, prefix + '1')
        return tree
    else:
        result[label] = prefix
        return label

def huffman_code(_vals):
    vals = _vals.copy()
    nodes = {}
    for n in vals.keys():  # leafs initialization
        nodes[n] = []

    while len(vals) > 1:  # binary tree creation
        s_vals = sorted(vals.items(), key=lambda x: x[1])
        a1 = s_vals[0][0]
        a2 = s_vals[1][0]
        vals[a1 + a2] = vals.pop(a1) + vals.pop(a2)
        nodes[a1 + a2] = [a1, a2]
    code = {}
    root = a1 + a2
    tree = {}
    tree = assign_code(nodes, root, code)
    return code, tree

def probability_dict(n_chars, chain):
    frecuency_lst = []
    for i in range(n_chars):
        frecuency_lst.append(chain.count(i))
    prob_lst = []
    for i in range(n_chars):
        prob_lst.append((frecuency_lst[i] / len(chain), str(i)))
    dic = {c: p for p, c in prob_lst}
    return dic, frecuency_lst

def huffman_entropy_codification(n_chars, chain):
    dic, fr_lst = probability_dict(n_chars, chain)
    code, tree = huffman_code(dic)
    cod = []
    huff = ""
    prom = 0
    for i in range(n_chars):
        cod.append(code[str(i)])
    
    for i in range(len(cod)):
        print(f"{i} -> {cod[i]}")
        prom += (fr_lst[i] / len(chain)) * len(cod[i])
    
    for i in range(len(chain)):
        huff += cod[chain[i]]
    
    return huff, prom

def encoder(singal, singal_dict):
    Low = 0
    High = 1
    for s in singal:
        CodeRange = High - Low
        High = Low + CodeRange * singal_dict[s][1]
        Low = Low + CodeRange * singal_dict[s][0]
    med=(High+Low)/2
    return med

def decoder(encoded_number, singal_dict, singal_length):
    singal = []
    while singal_length:
        for k, v in singal_dict.items():
            if v[0] <= encoded_number < v[1]:
                singal.append(k)
                range = v[1] - v[0]
                encoded_number -= v[0]
                encoded_number /= range
                break
        singal_length -= 1
    return singal

def aritmetic_codification(n_chars, chain):
    dic, fr_lst = probability_dict(n_chars, chain)
    ini = fin = 0
    signal = ""
    singal_dict = {}
    for i in range(len(fr_lst)):
        fin+=fr_lst[i]/len(chain)
        singal_dict[str(i)] = (ini,fin)
        ini=fin
    print(singal_dict)
    for i in chain:
        signal += str(i)
    arit_code = encoder(signal, singal_dict)
    decoded_chain = decoder(arit_code, singal_dict)
    return arit_code, decoded_chain

if __name__ == "__main__":
    # Read the image
    for i in range(1, 16):
        img_f, imgf4, imgpath = prepare_image(i)
        imgbytes = os.path.getsize(imgpath)
        # Create a image for the f4 chain code with the original image size
        img_f4 = np.zeros((imgf4.shape[0], imgf4.shape[1], 2), dtype=np.uint8)
        for row, col in np.ndindex(img_f.shape):
            img_f4[row, col] = np.array([imgf4[row, col], imgf4[row, col]])

        # Apply the freeman 4 chain code algorithm
        chc_f4 = freeman4ChainCode(img_f4)
        # print(f"Freeman 4 chain code: {chc_f4}")
        print(f"Freeman 4 chain code length: {len(chc_f4)}")
        print(f"Freeman 4 chain code entropy H = {calculate_entropy(4, chc_f4)}")
        huff_f4, prom = huffman_entropy_codification(4, chc_f4)
        print(f"Huffman entropy codification: {huff_f4}")
        print(f"Total bits: {len(huff_f4)}")
        print(f"Bits per pixel: {prom}")
        a_c_f4, d_c_f4 = aritmetic_codification(4, chc_f4)
        print(f"Aritmetic codification: {a_c_f4}")
        print(f"Decoded chain: {d_c_f4}")
        huff4bytes = len(huff_f4) / 8
        print("Compression ratio: {:0.2f}".format((1-(huff4bytes/imgbytes))*100))
        # Apply the freeman 8 chain code algorithm
        # chc_f8 = freeman8ChainCode(img_f)
        # print(f"Freeman 8 chain code: {chc_f8}")
        # print(f"Freeman 8 chain code length: {len(chc_f8)}")
        # print(f"Freeman 8 chain code entropy H = {calculate_entropy(8, chc_f8)}")
        # f8_prob = probability_dict(8, chc_f8)
        # cv.waitKey(0)
        # cv.destroyAllWindows()