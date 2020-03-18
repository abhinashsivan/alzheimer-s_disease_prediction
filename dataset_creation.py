import glob
import csv
import cv2
from skimage import feature
import numpy as np
import glob
import pandas as pd

count=0
for image in glob.glob('dataset_online/train/positive/*.jpg'):

    im = cv2.imread(image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.array(im, dtype=np.uint8)
    ngcm = feature.texture.greycomatrix(im, [1], [0], 256, symmetric=False, normed=True)


    contrast = feature.texture.greycoprops(ngcm, 'contrast')
    dissimilarity = feature.texture.greycoprops(ngcm, 'dissimilarity')
    homogeneity = feature.texture.greycoprops(ngcm, 'homogeneity')
    energy = feature.texture.greycoprops(ngcm, 'energy')
    correlation = feature.texture.greycoprops(ngcm, 'correlation')
    ASM = feature.texture.greycoprops(ngcm, 'ASM')

    ls=[contrast[0][0], dissimilarity[0][0], homogeneity[0][0], energy[0][0], correlation[0][0], ASM[0][0]]
    print(" yes - image %d done\n\n"%count)
    count+=1
    with open('yes.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(ls)

count=0
for image in glob.glob('dataset_online/train/negative/*.jpg'):
    im = cv2.imread(image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.array(im, dtype=np.uint8)
    ngcm = feature.texture.greycomatrix(im, [1], [0], 256, symmetric=False, normed=True)


    contrast = feature.texture.greycoprops(ngcm, 'contrast')
    dissimilarity = feature.texture.greycoprops(ngcm, 'dissimilarity')
    homogeneity = feature.texture.greycoprops(ngcm, 'homogeneity')
    energy = feature.texture.greycoprops(ngcm, 'energy')
    correlation = feature.texture.greycoprops(ngcm, 'correlation')
    ASM = feature.texture.greycoprops(ngcm, 'ASM')

    ls=[contrast[0][0], dissimilarity[0][0], homogeneity[0][0], energy[0][0], correlation[0][0], ASM[0][0]]
    print(" no - image %d done\n\n" %count)
    count+=1
    with open('no.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(ls)
df1=pd.read_csv("yes.csv")
df2=pd.read_csv("no.csv")


df1.columns=range(7)
df2.columns=range(7)

df=pd.concat([df1,df2],axis=0)

df.to_csv('m.csv')
