import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os import remove
#Codificacion Huffman

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


def Huffman_code(_vals):
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
    tree = assign_code(nodes, root, code)  # assignment of the code for the given binary tree
    return code, tree

#Codificacion Aritmetrica

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

#Tablas de convercion

f4ToVcc=[[1  ,2  ,"*",0  ],
         [0  ,1  ,2  ,"*"],
         ["*",0  ,1  ,2  ],
         [2  ,"*",0  ,1  ]]

f4To3Ot=[[1  ,"*",2  ,"*"],
         ["*",1  ,"*",2  ],
         [2  ,"*",1  ,"*"],
         ["*",2  ,"*",1  ]]

f8ToAf8=[[0,1,3,5,7,6,4,2],
         [2,0,1,3,5,7,6,4],
         [4,2,0,1,3,5,7,6],
         [6,4,2,0,1,3,5,7],
         [7,6,4,2,0,1,3,5],
         [5,7,6,4,2,0,1,3],
         [3,5,7,6,4,2,0,1],
         [1,3,5,7,6,4,2,0]]

byteshufff4=[]
byteshuffvcc=[]
byteshuff3ot=[]
byteshufff8=[]
byteshuffaf8=[]
#Imagen a procesar

#imagen = cv2.imread('1.bmp')
'''imagen1=[[[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]],
        [[255,255,255],[255,255,255],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[255,255,255],[255,255,255],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[255,255,255]],
        [[255,255,255],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[255,255,255],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[255,255,255],[255,255,255],[ 0 , 0 , 0 ],[255,255,255]],
        [[255,255,255],[255,255,255],[ 0 , 0 , 0 ],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[ 0 , 0 , 0 ],[255,255,255],[255,255,255]],
        [[255,255,255],[255,255,255],[ 0 , 0 , 0 ],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[ 0 , 0 , 0 ],[255,255,255]],
        [[255,255,255],[ 0 , 0 , 0 ],[255,255,255],[255,255,255],[255,255,255],[ 0 , 0 , 0 ],[255,255,255],[255,255,255],[ 0 , 0 , 0 ],[255,255,255]],
        [[255,255,255],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[255,255,255],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[ 0 , 0 , 0 ],[255,255,255]],
        [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]]]
imagen2=[[[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]],
        [[255,255,255],[255,255,255],[0,0,0]  ,[0,0,0]  ,[0,0,0]  ,[0,0,0]  ,[255,255,255],[255,255,255]],
        [[255,255,255],[0,0,0]  ,[255,255,255],[255,255,255],[255,255,255],[0,0,0]  ,[255,255,255],[255,255,255]],
        [[255,255,255],[0,0,0]  ,[0,0,0]  ,[0,0,0]  ,[0,0,0]  ,[255,255,255],[255,255,255],[255,255,255]],
        [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]]]'''

imagen=""

for numImg in range(1,16):

    '''if numImg==1:
        imagen=np.copy(imagen1)
    elif numImg==2:
        imagen=np.copy(imagen2)'''

    imide=str(numImg)+'.bmp'

    imagen = cv2.imread(imide)

    nomImg=str(numImg)+'.bmp'

    plt.figure("Original")
    plt.imshow(imagen)
    imagenaux=np.copy(imagen)
    for x in range(1,len(imagen)-1):
        for y in range(1,len(imagen[x])-1):
            su=sum((imagen[x - 1][y - 1][0], imagen[x][y - 1][0], imagen[x + 1][y - 1][0], imagen[x - 1][y][0],
                imagen[x + 1][y][0], imagen[x - 1][y + 1][0], imagen[x][y + 1][0], imagen[x + 1][y + 1][0]))
            if su==0 or su==1785:
                imagenaux[x][y]= 255
    for x in range(1,len(imagenaux)-1):
        for y in range(1,len(imagenaux[x])-1):
            su=sum((imagenaux[x - 1][y - 1][0], imagenaux[x][y - 1][0], imagenaux[x + 1][y - 1][0], imagenaux[x - 1][y][0],
                imagenaux[x + 1][y][0], imagenaux[x - 1][y + 1][0], imagenaux[x][y + 1][0], imagenaux[x + 1][y + 1][0]))
            if su==0 or su==1785:
                imagenaux[x][y]= 255
    for x in range(1,len(imagenaux)-1):
        for y in range(1,len(imagenaux[x])-1):
            if imagenaux[x][y][0]==0:
                ejex=x
                ejey=y
                break
            else:
                continue
        break
    imagenaux[ejex][ejey][0]=255
    plt.figure("Bordes")
    plt.imshow(imagenaux)

    ex=ejex
    ey=ejey

    #Calcular Codigo F4

    aux=[]
    del aux[:]
    f4=[]
    del f4[:]
    band=2
    band2=0
    if imagenaux[ejex][ejey - 1][1] != 0:
        aux.append(1)
        if imagenaux[ejex - 1][ejey - 1][1] != 0:
            if imagenaux[ejex - 1][ejey][1] != 0:
                f4.append(0)
                if imagenaux[ejex - 1][ejey + 1][1] != 0:
                    if imagenaux[ejex][ejey + 1][1] != 0:
                        f4.append(3)

    while True:

        if imagenaux[ejex][ejey+1][0]==0:
            imagenaux[ejex][ejey+1][0] = 255
            ejey += 1
            if imagenaux[ejex - 1][ejey][1] != 0:
                f4.append(0)
                if imagenaux[ejex - 1][ejey + 1][1] != 0:
                    if imagenaux[ejex][ejey + 1][1] != 0:
                        f4.append(3)
                        if imagenaux[ejex + 1][ejey + 1][1] != 0:
                            if imagenaux[ejex + 1][ejey][1] != 0:
                                f4.append(2)

        elif imagenaux[ejex+1][ejey][0]==0:
            imagenaux[ejex+1][ejey][0] = 255
            ejex += 1
            if imagenaux[ejex][ejey + 1][1] != 0:
                f4.append(3)
                if imagenaux[ejex + 1][ejey + 1][1] != 0:
                    if imagenaux[ejex + 1][ejey][1] != 0:
                        f4.append(2)
                        if imagenaux[ejex + 1][ejey - 1][1] != 0:
                            if imagenaux[ejex][ejey-1][1] != 0:
                                f4.append(1)

        elif imagenaux[ejex][ejey-1][0]==0:
            imagenaux[ejex][ejey-1][0] = 255
            ejey -= 1
            if imagenaux[ejex + 1][ejey][1] != 0:
                f4.append(2)
                if imagenaux[ejex + 1][ejey - 1][1] != 0:
                    if imagenaux[ejex][ejey-1][1] != 0:
                        f4.append(1)
                        if imagenaux[ejex - 1][ejey - 1][1] != 0:
                            if imagenaux[ejex-1][ejey][1] != 0:
                                f4.append(0)

        elif imagenaux[ejex-1][ejey][0]==0:
            imagenaux[ejex-1][ejey][0] = 255
            ejex-=1
            if imagenaux[ejex][ejey - 1][1] != 0:
                f4.append(1)
                if imagenaux[ejex - 1][ejey - 1][1] != 0:
                    if imagenaux[ejex - 1][ejey][1] != 0:
                        f4.append(0)
                        if imagenaux[ejex - 1][ejey + 1][1] != 0:
                            if imagenaux[ejex][ejey+1][1] != 0:
                                f4.append(3)

        elif imagenaux[ejex-1][ejey+1][0]==0:
            imagenaux[ejex-1][ejey+1][0] = 255
            ejex -= 1
            ejey += 1
            if imagenaux[ejex][ejey - 1][1] != 0:
                f4.append(1)
                if imagenaux[ejex - 1][ejey - 1][1] != 0:
                    if imagenaux[ejex - 1][ejey][1] != 0:
                        f4.append(0)
                        if imagenaux[ejex - 1][ejey + 1][1] != 0:
                            if imagenaux[ejex][ejey + 1][1] != 0:
                                f4.append(3)
                                if imagenaux[ejex + 1][ejey + 1][1] != 0:
                                    if imagenaux[ejex + 1][ejey][1] != 0:
                                        f4.append(2)

        elif imagenaux[ejex+1][ejey+1][0]==0:
            imagenaux[ejex+1][ejey+1][0] = 255
            ejex += 1
            ejey += 1
            if imagenaux[ejex - 1][ejey][1] != 0:
                f4.append(0)
                if imagenaux[ejex - 1][ejey + 1][1] != 0:
                    if imagenaux[ejex][ejey + 1][1] != 0:
                        f4.append(3)
                        if imagenaux[ejex + 1][ejey + 1][1] != 0:
                            if imagenaux[ejex + 1][ejey][1] != 0:
                                f4.append(2)
                                if imagenaux[ejex + 1][ejey - 1][1] != 0:
                                    if imagenaux[ejex][ejey-1][1] != 0:
                                        f4.append(1)

        elif imagenaux[ejex+1][ejey-1][0]==0:
            imagenaux[ejex+1][ejey-1][0] = 255
            ejex += 1
            ejey -= 1
            if imagenaux[ejex][ejey + 1][1] != 0:
                f4.append(3)
                if imagenaux[ejex + 1][ejey + 1][1] != 0:
                    if imagenaux[ejex + 1][ejey][1] != 0:
                        f4.append(2)
                        if imagenaux[ejex + 1][ejey - 1][1] != 0:
                            if imagenaux[ejex][ejey-1][1] != 0:
                                f4.append(1)
                                if imagenaux[ejex - 1][ejey - 1][1] != 0:
                                    if imagenaux[ejex-1][ejey][1] != 0:
                                        f4.append(0)

        elif imagenaux[ejex-1][ejey-1][0]==0:
            imagenaux[ejex-1][ejey-1][0] = 255
            ejex -= 1
            ejey -= 1
            if imagenaux[ejex + 1][ejey][1] != 0:
                f4.append(2)
                if imagenaux[ejex + 1][ejey - 1][1] != 0:
                    if imagenaux[ejex][ejey - 1][1] != 0:
                        f4.append(1)
                        if imagenaux[ejex - 1][ejey - 1][1] != 0:
                            if imagenaux[ejex - 1][ejey][1] != 0:
                                f4.append(0)
                                if imagenaux[ejex - 1][ejey + 1][1] != 0:
                                    if imagenaux[ejex][ejey+1][1] != 0:
                                        f4.append(3)
        else:
            break
    while len(aux)>0:
        f4.append(aux[0])
        aux.pop()

    nomTxt='datos/datos de '+nomImg+'.txt'
    nomTxtCod='codigos/f4/f4 de '+nomImg+'.txt'

    file = open(nomTxt, 'a')
    file.close()
    remove(nomTxt)
    file = open(nomTxtCod, 'a')
    file.close()
    remove(nomTxtCod)

    saveFile = open(nomTxt, 'a')
    saveFileCod = open(nomTxtCod, 'a')

    print("F4 = ",f4)
    for x in f4:
        saveFileCod.write(str(x))
    saveFileCod.close()
    saveFile.write("F4 = "+str(f4)+"\n")
    lonf4=len(f4)
    print("Longitud de la cadena: ",lonf4)
    saveFile.write("Longitud de la cadena: "+str(lonf4)+"\n")
    f4bit=[0,0,0,0]
    for x in range(len(f4)):
        if f4[x] == 0:
            f4bit[0] += 1
        if f4[x] == 1:
            f4bit[1] += 1
        if f4[x] == 2:
            f4bit[2] += 1
        if f4[x] == 3:
            f4bit[3] += 1

    f4H=0
    for x in range(4):
        if f4bit[x]!=0:
            f4H = (-(((f4bit[x]) / lonf4) * (np.log2(f4bit[x] / lonf4))))+f4H
    f4H='{:0.4f}'.format(f4H)
    print("H = ",f4H)
    saveFile.write("H = "+str(f4H) + "\n")

    freqf4 = []
    del freqf4[:]

    for x in range(len(f4bit)):
        proba=f4bit[x]/lonf4
        freqf4.append((proba,str(x)))

    vals = {l:v for (v,l) in freqf4}
    codef4, treef4 = Huffman_code(vals)

    codf4=[]
    del codf4[:]

    for x in range(len(f4bit)):
        text=str(x)
        encoded = codef4[text]
        codf4.append(encoded)

    xprom=0

    for x in range(len(codf4)):
        print(x,": ",codf4[x])
        saveFile.write(str(x)+": "+str(codf4[x]) + "\n")
        xprom+=((f4bit[x]/lonf4)*(len(codf4[x])))

    huff4=""
    for x in range(len(f4)):
        huff4+=codf4[f4[x]]

    print("Cod Huffman: ",huff4," Total de bits= ",len(huff4))
    saveFile.write("Cod Huffman: "+str(huff4)+" Total de bits= "+str(len(huff4)) + "\n")
    print("x̄=",'{:0.4f}'.format(xprom)," bits/pixel")
    escaux="xprom="+str('{:0.4f}'.format(xprom))+" bits/pixel" + "\n"
    saveFile.write(escaux)

    ini=0
    fin=0
    singal_dictf4 = {}
    singal_dictf4.clear()
    for x in range(len(f4bit)):
        fin+=f4bit[x]/lonf4
        singal_dictf4[str(x)] = (ini,fin)
        ini=fin
    print(singal_dictf4)
    saveFile.write(str(singal_dictf4) + "\n")

    singalf4=""

    for x in range(len(f4)):
        singalf4+=str(f4[x])

    ans = encoder(singalf4, singal_dictf4)
    print("Codigo Aritmetico = ",ans)
    saveFile.write("Codigo Aritmetico = "+str(ans) + "\n")

    singal_recf4 = decoder(ans, singal_dictf4, len(singalf4))
    print("Cadena Decodificada",singal_recf4)
    saveFile.write("Cadena Decodificada"+str(singal_recf4) + "\n")

    img = np.array(imagen)
    cv2.imwrite('imgToPng.png',img)
    pngbytes=os.path.getsize('imgToPng.png')
    huffbytes=len(huff4)/8
    rc='{:0.2f}'.format((1-(huffbytes/pngbytes))*100)
    print("RC respecto a PNG= ",rc,"%")
    saveFile.write("RC respecto a PNG= "+str(rc)+"%" + "\n\n")

    print()


    f8=[]
    del f8[:]
    ba=0
    ejex=ex
    ejey=ey
    imagenaux[ex][ey][0] = 0

    while True:
        if imagenaux[ejex][ejey+1][0]==255 and imagenaux[ejex][ejey+1][1]==0:
            imagenaux[ejex][ejey+1][0] = 0
            ejey += 1
            f8.append(0)
        elif imagenaux[ejex+1][ejey][0]==255 and imagenaux[ejex+1][ejey][1]==0:
            imagenaux[ejex+1][ejey][0] = 0
            ejex += 1
            f8.append(6)
        elif imagenaux[ejex][ejey-1][0]==255 and imagenaux[ejex][ejey-1][1]==0:
            imagenaux[ejex][ejey-1][0] = 0
            ejey -= 1
            f8.append(4)
        elif imagenaux[ejex-1][ejey][0]==255 and imagenaux[ejex-1][ejey][1]==0:
            imagenaux[ejex-1][ejey][0] = 0
            ejex-=1
            f8.append(2)
        elif imagenaux[ejex-1][ejey+1][0]==255 and imagenaux[ejex-1][ejey+1][1]==0:
            imagenaux[ejex-1][ejey+1][0] = 0
            ejex -= 1
            ejey += 1
            f8.append(1)
        elif imagenaux[ejex+1][ejey+1][0]==255 and imagenaux[ejex+1][ejey+1][1]==0:
            imagenaux[ejex+1][ejey+1][0] = 0
            ejex += 1
            ejey += 1
            f8.append(7)
        elif imagenaux[ejex+1][ejey-1][0]==255 and imagenaux[ejex+1][ejey-1][1]==0:
            imagenaux[ejex+1][ejey-1][0] = 0
            ejex += 1
            ejey -= 1
            f8.append(5)
        elif imagenaux[ejex-1][ejey-1][0]==255 and imagenaux[ejex-1][ejey-1][1]==0:
            imagenaux[ejex-1][ejey-1][0] = 0
            ejex -= 1
            ejey -= 1
            f8.append(3)
        else:
            if ba==0:
                imagenaux[ex][ey][0]=255
                ba=1
            else:
                break

    #F4_a_VCC

    vcc=[]
    del vcc[:]

    for x in range(len(f4)-1):
        vcc.append(f4ToVcc[f4[x]][f4[x+1]])
    vcc.append(0)

    nomTxtCod = 'codigos/vcc/vcc de ' + nomImg + '.txt'
    file = open(nomTxtCod, 'a')
    file.close()
    remove(nomTxtCod)
    saveFileCod = open(nomTxtCod, 'a')

    print("VCC = ",vcc)
    saveFile.write("VCC = "+str(vcc) + "\n")
    for x in vcc:
        saveFileCod.write(str(x))
    saveFileCod.close()
    lonvcc=len(vcc)
    print("Longitud de la cadena: ",lonvcc)
    saveFile.write("Longitud de la cadena: "+str(lonvcc) + "\n")

    vccbit=[0,0,0]
    for x in range(len(vcc)):
        if vcc[x] == 0:
            vccbit[0] += 1
        if vcc[x] == 1:
            vccbit[1] += 1
        if vcc[x] == 2:
            vccbit[2] += 1

    vccH=0
    for x in range(3):
        if vccbit[x]!=0:
            vccH = (-(((vccbit[x]) / lonvcc) * (np.log2(vccbit[x] / lonvcc))))+vccH
    vccH='{:0.4f}'.format(vccH)
    print("H = ",vccH)
    saveFile.write("H = "+str(vccH) + "\n")

    freqvcc = []
    del freqvcc[:]

    for x in range(len(vccbit)):
        proba=vccbit[x]/lonvcc
        freqvcc.append((proba,str(x)))

    vals = {l:v for (v,l) in freqvcc}
    codevcc, treevcc = Huffman_code(vals)

    codvcc=[]
    del codvcc[:]

    for x in range(len(vccbit)):
        text=str(x)
        encoded = codevcc[text]
        codvcc.append(encoded)

    xprom=0

    for x in range(len(codvcc)):
        print(x,": ",codvcc[x])
        saveFile.write(str(x)+": "+str(codvcc[x]) + "\n")
        xprom+=((vccbit[x]/lonvcc)*(len(codvcc[x])))

    hufvcc=""
    for x in range(len(vcc)):
        hufvcc+=codvcc[vcc[x]]

    print("Cod Huffman: ",hufvcc," Total de bits= ",len(hufvcc))
    saveFile.write("Cod Huffman: "+str(hufvcc)+" Total de bits= "+str(len(hufvcc)) + "\n")
    print("x̄=",'{:0.4f}'.format(xprom)," bits/pixel")
    saveFile.write("xprom="+str('{:0.4f}'.format(xprom))+" bits/pixel"+ "\n")

    ini=0
    fin=0
    singal_dictvcc = {}
    singal_dictvcc.clear()
    for x in range(len(vccbit)):
        fin+=vccbit[x]/lonvcc
        singal_dictvcc[str(x)] = (ini,fin)
        ini=fin
    print(singal_dictvcc)
    saveFile.write(str(singal_dictvcc) + "\n")

    singalvcc=""

    for x in range(len(vcc)):
        singalvcc+=str(vcc[x])

    ans = encoder(singalvcc, singal_dictvcc)
    print("Codigo Aritmetico = ",ans)
    saveFile.write("Codigo Aritmetico = "+str(ans) + "\n")

    singal_recvcc = decoder(ans, singal_dictvcc, len(singalvcc))
    print("Cadena Decodificada",singal_recvcc)
    saveFile.write("Cadena Decodificada"+str(singal_recvcc) + "\n")

    huffbytes=len(hufvcc)/8
    rc='{:0.2f}'.format((1-(huffbytes/pngbytes))*100)
    print("RC respecto a PNG= ",rc,"%")
    saveFile.write("RC respecto a PNG= "+str(rc)+"%" + "\n\n")

    print()

    #F4_a_3OT

    tres0t=[]
    del tres0t[:]
    auxt=1

    for x in range(len(f4)-1):
        if f4[x]==f4[x+1]:
            tres0t.append(0)
        else:
            tres0t.append(f4To3Ot[auxt][f4[x+1]])
            auxt=f4[x]
    tres0t.append(1)

    nomTxtCod = 'codigos/3ot/3ot de ' + nomImg + '.txt'
    file = open(nomTxtCod, 'a')
    file.close()
    remove(nomTxtCod)
    saveFileCod = open(nomTxtCod, 'a')

    print("3OT = ",tres0t)
    saveFile.write("3OT = "+str(tres0t) + "\n")
    for x in tres0t:
        saveFileCod.write(str(x))
    saveFileCod.close()
    lon3ot=len(tres0t)
    print("Longitud de la cadena: ",lon3ot)
    saveFile.write("Longitud de la cadena: "+str(lon3ot) + "\n")

    tres0tbit=[0,0,0]
    for x in range(len(tres0t)):
        if tres0t[x] == 0:
            tres0tbit[0] += 1
        if tres0t[x] == 1:
            tres0tbit[1] += 1
        if tres0t[x] == 2:
            tres0tbit[2] += 1

    tres0tH=0
    for x in range(3):
        if tres0tbit[x]!=0:
            tres0tH = (-(((tres0tbit[x]) / lon3ot) * (np.log2(tres0tbit[x] / lon3ot))))+tres0tH
    tres0tH='{:0.4f}'.format(tres0tH)
    print("H = ",tres0tH)
    saveFile.write("H = "+str(tres0tH) + "\n")

    freqtres0t = []
    del freqtres0t[:]

    for x in range(len(tres0tbit)):
        proba=tres0tbit[x]/lon3ot
        freqtres0t.append((proba,str(x)))

    vals = {l:v for (v,l) in freqvcc}
    codetres0t, treetres0t = Huffman_code(vals)

    codtres0t=[]
    del codtres0t[:]

    for x in range(len(tres0tbit)):
        text=str(x)
        encoded = codetres0t[text]
        codtres0t.append(encoded)

    xprom=0

    for x in range(len(codtres0t)):
        print(x,": ",codtres0t[x])
        saveFile.write(str(x)+": "+str(codtres0t[x]) + "\n")
        xprom+=((tres0tbit[x]/lon3ot)*(len(codtres0t[x])))

    huftres0t=""
    for x in range(len(tres0t)):
        huftres0t+=codtres0t[tres0t[x]]

    print("Cod Huffman: ",huftres0t," Total de bits= ",len(huftres0t))
    saveFile.write("Cod Huffman: "+str(huftres0t)+" Total de bits= "+str(len(huftres0t)) + "\n")
    print("x̄=",'{:0.4f}'.format(xprom)," bits/pixel")
    saveFile.write("xprom="+str('{:0.4f}'.format(xprom))+" bits/pixel" + "\n")

    ini=0
    fin=0
    singal_dicttres0t = {}
    singal_dicttres0t.clear()
    for x in range(len(tres0tbit)):
        fin+=tres0tbit[x]/lon3ot
        singal_dicttres0t[str(x)] = (ini,fin)
        ini=fin
    print(singal_dicttres0t)
    saveFile.write(str(singal_dicttres0t) + "\n")

    singaltres0t=""

    for x in range(len(tres0t)):
        singaltres0t+=str(tres0t[x])

    ans = encoder(singaltres0t, singal_dicttres0t)
    print("Codigo Aritmetico = ",ans)
    saveFile.write("Codigo Aritmetico = "+str(ans) + "\n")

    singal_rectres0t = decoder(ans, singal_dicttres0t, len(singaltres0t))
    print("Cadena Decodificada",singal_rectres0t)
    saveFile.write("Cadena Decodificada"+str(singal_rectres0t) + "\n")

    huffbytes=len(huftres0t)/8
    rc='{:0.2f}'.format((1-(huffbytes/pngbytes))*100)
    print("RC respecto a PNG= ",rc,"%")
    saveFile.write("RC respecto a PNG= "+str(rc)+"%" + "\n\n")

    print()

    #F8 a AF8

    Af8=[]
    del Af8[:]

    for x in range(len(f8)-1):
        Af8.append(f8ToAf8[f8[x]][f8[x+1]])
    Af8.append(2)

    nomTxtCod = 'codigos/f8/f8 de ' + nomImg + '.txt'
    file = open(nomTxtCod, 'a')
    file.close()
    remove(nomTxtCod)
    saveFileCod = open(nomTxtCod, 'a')

    print("F8 = ",f8)
    saveFile.write("F8 = "+str(f8) + "\n")
    for x in f8:
        saveFileCod.write(str(x))
    saveFileCod.close()
    lonf8=len(f8)
    print("Longitud de la cadena: ",lonf8)
    saveFile.write("Longitud de la cadena: "+str(lonf8) + "\n")
    f8bit=[0,0,0,0,0,0,0,0]
    for x in range(len(f8)):
        if f8[x] == 0:
            f8bit[0] += 1
        if f8[x] == 1:
            f8bit[1] += 1
        if f8[x] == 2:
            f8bit[2] += 1
        if f8[x] == 3:
            f8bit[3] += 1
        if f8[x] == 4:
            f8bit[4] += 1
        if f8[x] == 5:
            f8bit[5] += 1
        if f8[x] == 6:
            f8bit[6] += 1
        if f8[x] == 7:
            f8bit[7] += 1

    f8H=0
    for x in range(8):
        if f8bit[x]!=0:
            f8H = (-(((f8bit[x]) / lonf8) * (np.log2(f8bit[x] / lonf8))))+f8H
    f8H='{:0.4f}'.format(f8H)

    print("H = ",f8H)
    saveFile.write("H = "+str(f8H) + "\n")

    freqf8 = []
    del freqf8[:]

    for x in range(len(f8bit)):
        proba=f8bit[x]/lonf8
        freqf8.append((proba,str(x)))

    vals = {l:v for (v,l) in freqf8}
    codef8, treef8 = Huffman_code(vals)

    codf8=[]
    del codf8[:]

    for x in range(len(f8bit)):
        text=str(x)
        encoded = codef8[text]
        codf8.append(encoded)

    xprom=0

    for x in range(len(codf8)):
        print(x,": ",codf8[x])
        saveFile.write(str(x)+": "+str(codf8[x]) + "\n")
        xprom+=((f8bit[x]/lonf8)*(len(codf8[x])))

    huff8=""
    for x in range(len(f8)):
        huff8+=codf8[f8[x]]

    print("Cod Huffman: ",huff8," Total de bits= ",len(huff8))
    saveFile.write("Cod Huffman: "+str(huff8)+" Total de bits= "+str(len(huff8)) + "\n")
    print("x̄=",'{:0.4f}'.format(xprom)," bits/pixel")
    saveFile.write("xprom="+str('{:0.4f}'.format(xprom))+" bits/pixel" + "\n")

    ini=0
    fin=0
    singal_dictf8 = {}
    singal_dictf8.clear()
    for x in range(len(f8bit)):
        fin+=f8bit[x]/lonf8
        singal_dictf8[str(x)] = (ini,fin)
        ini=fin
    print(singal_dictf8)
    saveFile.write(str(singal_dictf8) + "\n")

    singalf8=""

    for x in range(len(f8)):
        singalf8+=str(f8[x])

    ans = encoder(singalf8, singal_dictf8)
    print("Codigo Aritmetico = ",ans)
    saveFile.write("Codigo Aritmetico = "+str(ans) + "\n")

    singal_recf8 = decoder(ans, singal_dictf8, len(singalf8))
    print("Cadena Decodificada",singal_recf8)
    saveFile.write("Cadena Decodificada"+str(singal_recf8) + "\n")

    huffbytes=len(huff8)/8
    rc='{:0.2f}'.format((1-(huffbytes/pngbytes))*100)
    print("RC respecto a PNG= ",rc,"%")
    saveFile.write("RC respecto a PNG= "+str(rc)+"%" + "\n\n")

    print()

    #AF8

    nomTxtCod = 'codigos/af8/af8 de ' + nomImg + '.txt'
    file = open(nomTxtCod, 'a')
    file.close()
    remove(nomTxtCod)
    saveFileCod = open(nomTxtCod, 'a')

    print("AF8 = ",Af8)
    saveFile.write("AF8 = "+str(Af8) + "\n")
    for x in Af8:
        saveFileCod.write(str(x))
    saveFileCod.close()
    lonaf8=len(Af8)
    print("Longitud de la cadena: ",lonaf8)
    saveFile.write("Longitud de la cadena: "+str(lonaf8) + "\n")
    Af8bit=[0,0,0,0,0,0,0,0]
    for x in range(len(Af8)):
        if Af8[x] == 0:
            Af8bit[0] += 1
        if Af8[x] == 1:
            Af8bit[1] += 1
        if Af8[x] == 2:
            Af8bit[2] += 1
        if Af8[x] == 3:
            Af8bit[3] += 1
        if Af8[x] == 4:
            Af8bit[4] += 1
        if Af8[x] == 5:
            Af8bit[5] += 1
        if Af8[x] == 6:
            Af8bit[6] += 1
        if Af8[x] == 7:
            Af8bit[7] += 1

    Af8H=0
    for x in range(8):
        if Af8bit[x]!=0:
            Af8H = (-(((Af8bit[x]) / lonaf8) * (np.log2(Af8bit[x] / lonaf8))))+Af8H
    Af8H='{:0.4f}'.format(Af8H)

    print("H = ",Af8H)
    saveFile.write("H = "+str(Af8H) + "\n")

    freqAf8 = []
    del freqAf8[:]

    for x in range(len(Af8bit)):
        proba=Af8bit[x]/lonaf8
        freqAf8.append((proba,str(x)))

    vals = {l:v for (v,l) in freqAf8}
    codeAf8, treeAf8 = Huffman_code(vals)

    codAf8=[]
    del codAf8[:]

    for x in range(len(Af8bit)):
        text=str(x)
        encoded = codeAf8[text]
        codAf8.append(encoded)

    xprom=0

    for x in range(len(codAf8)):
        print(x,": ",codAf8[x])
        saveFile.write(str(x)+": "+str(codAf8[x]) + "\n")
        xprom+=((Af8bit[x]/lonaf8)*(len(codAf8[x])))

    hufAf8=""
    for x in range(len(Af8)):
        hufAf8+=codAf8[Af8[x]]

    print("Cod Huffman: ",hufAf8," Total de bits= ",len(hufAf8))
    saveFile.write("Cod Huffman: "+str(hufAf8)+" Total de bits= "+str(len(hufAf8)) + "\n")
    print("x̄=",'{:0.4f}'.format(xprom)," bits/pixel")
    saveFile.write("xprom="+str('{:0.4f}'.format(xprom))+" bits/pixel" + "\n")

    ini=0
    fin=0
    singal_dictAf8 = {}
    singal_dictAf8.clear()
    for x in range(len(Af8bit)):
        fin+=Af8bit[x]/lonaf8
        singal_dictAf8[str(x)] = (ini,fin)
        ini=fin
    print(singal_dictAf8)
    saveFile.write(str(singal_dictAf8) + "\n")

    singalAf8=""

    for x in range(len(Af8)):
        singalAf8+=str(Af8[x])

    ans = encoder(singalAf8, singal_dictAf8)
    print("Codigo Aritmetico = ",ans)
    saveFile.write("Codigo Aritmetico = "+str(ans) + "\n")

    singal_recAf8 = decoder(ans, singal_dictAf8, len(singalAf8))
    print("Cadena Decodificada",singal_recAf8)
    saveFile.write("Cadena Decodificada"+str(singal_recAf8) + "\n")

    huffbytes=len(hufAf8)/8
    rc='{:0.2f}'.format((1-(huffbytes/pngbytes))*100)
    print("RC respecto a PNG= ",rc,"%")
    saveFile.write("RC respecto a PNG= "+str(rc)+"%" + "\n")

    print()
    saveFile.close()
    byteshufff4.append((len(huff4)/8))
    byteshuffvcc.append((len(hufvcc) / 8))
    byteshuff3ot.append((len(huftres0t) / 8))
    byteshufff8.append((len(huff8) / 8))
    byteshuffaf8.append((len(hufAf8) / 8))

plt.show()

print(byteshufff4)
print(byteshuffvcc)
print(byteshuff3ot)
print(byteshufff8)
print(byteshuffaf8)

def graficar(datosx,datosy,z):
    nom="Promedio global "+z
    plt.figure(nom)
    plt.bar(datosx,datosy)
    plt.title('Promedio global '+z)
    plt.xlabel('Codigo de cadena')
    plt.ylabel('Promedio')
    plt.savefig("graficas/Grafica "+z+".png")
    return None

promf4=[0,0,0,0]
for x in range(1,16):
    nomtxt=nomTxtCod='codigos/f4/f4 de '+str(x)+'.bmp.txt'
    file=open(nomtxt,'r')
    for linea in file:
        cadena=linea
    for y in range(4):
        for w in range(len(cadena)):
            if int(cadena[w])==y:
                promf4[y]+=1
file.close()
for x in range(4):
    promf4[x]/=15

promvcc=[0,0,0]
for x in range(1,16):
    nomtxt=nomTxtCod='codigos/vcc/vcc de '+str(x)+'.bmp.txt'
    file=open(nomtxt,'r')
    for linea in file:
        cadena=linea
    for y in range(3):
        for w in range(len(cadena)):
            if int(cadena[w])==y:
                promvcc[y]+=1
file.close()
for x in range(3):
    promvcc[x]/=15

prom3ot=[0,0,0]
for x in range(1,16):
    nomtxt=nomTxtCod='codigos/3ot/3ot de '+str(x)+'.bmp.txt'
    file=open(nomtxt,'r')
    for linea in file:
        cadena=linea
    for y in range(3):
        for w in range(len(cadena)):
            if int(cadena[w])==y:
                prom3ot[y]+=1
file.close()
for x in range(3):
    prom3ot[x]/=15

promf8=[0,0,0,0,0,0,0,0]
for x in range(1,16):
    nomtxt=nomTxtCod='codigos/f8/f8 de '+str(x)+'.bmp.txt'
    file=open(nomtxt,'r')
    for linea in file:
        cadena=linea
    for y in range(8):
        for w in range(len(cadena)):
            if int(cadena[w])==y:
                promf8[y]+=1
file.close()
for x in range(8):
    promf8[x]/=15

promaf8=[0,0,0,0,0,0,0,0]
for x in range(1,16):
    nomtxt=nomTxtCod='codigos/af8/af8 de '+str(x)+'.bmp.txt'
    file=open(nomtxt,'r')
    for linea in file:
        cadena=linea
    for y in range(8):
        for w in range(len(cadena)):
            if int(cadena[w])==y:
                promaf8[y]+=1
file.close()
for x in range(8):
    promaf8[x]/=15

graficar(['0','1','2','3'],promf4,'F4')
graficar(['0','1','2'],promvcc,'VCC')
graficar(['0','1','2'],prom3ot,'3OT')
graficar(['0','1','2','3','4','5','6','7'],promf8,'F8')
graficar(['0','1','2','3','4','5','6','7'],promaf8,'AF8')

#plt.show()
