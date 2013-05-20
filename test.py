#!/usr/bin/python2

from network import network
import Image
import cv
import cv2
import numpy as np
import pprint

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print "hist_lines applicable only for grayscale images"
        #print "so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return hist

pp = pprint.PrettyPrinter(indent=4)

image_index = 1
samples 	= 7
truth_in 	= []
img 		= []
hist 		= []

for h in xrange(0,samples):
	truth_in.append([])
	img.append([])
	img[h] 	= cv.LoadImage(str(h+1)+'.jpg')
	gray 	= cv.CreateImage(cv.GetSize(img[h]), cv.IPL_DEPTH_8U, 1)
	cv.CvtColor(img[h], gray, cv.CV_RGB2GRAY)
	img[h] 	= np.asarray(gray[:,:])
	hist.append([hist_lines(img[h])])
	hist[h] = hist[h][0]




net = network(len(hist[0]),1,50,3) 					#inputs, hidden_layers, hidden_neurons, outputs

truth_in			= net.scale(255,hist)
print truth_in[1]
net.loadWeights("comp_gen_dice1.txt")

net.calcOuts(truth_in[0])

print net.outs[2]