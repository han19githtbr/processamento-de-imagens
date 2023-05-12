import numpy
import cv2
import mahotas

def print_image_wait(label, img):
    cv2.imshow(label, img)
    cv2.waitKey(0)

def BinarizationImg(img):
    T = mahotas.thresholding.otsu(img)
    bin = img.copy()
    bin[bin > T] = 255
    bin[bin < 255] = 0
    return cv2.bitwise_not(bin)

def draw_contours(img, objects):
    objects_length = len(objects)
    for c in range(objects_length):
        cv2.drawContours(img, objects,c,(0,128,0),2)


#open target image
target_img = cv2.imread('asserts/alpha2.png')
#open alphabet image
alpha_img = cv2.imread('asserts/alpha2.png')

target_img = cv2.resize(target_img, (0,0), fx = 2.0, fy = 2.0)

#gray scale convert images
target_img_grayscale = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
alpha_img_grayscale = cv2.cvtColor(alpha_img, cv2.COLOR_BGR2GRAY)

cv2.imshow('GrayScaler', target_img_grayscale)
cv2.waitKey(0)

#Step 2: Bilateral filter
target_smooths = cv2.bilateralFilter(target_img_grayscale, 10, 75, 75)
alpha_smooths = cv2.bilateralFilter(alpha_img_grayscale, 10, 75, 75)

cv2.imshow('Bilateral filter', target_smooths)
cv2.waitKey(0)
#Step 3: Binarization resulting in white and black pixels

target_bin = BinarizationImg(target_smooths)
alpha_bin = BinarizationImg(alpha_smooths)

cv2.imshow('Binarization', target_bin)
cv2.waitKey(0)


#Step 4: Detect edges using Canny
target_edges = cv2.Canny(target_bin, 70, 150)
alpha_edges = cv2.Canny(alpha_bin, 70, 150)

cv2.imshow('Canny', target_edges)
cv2.waitKey(0)


#Step 5: Identify contours

target_contours, target_hierarchy = cv2.findContours(target_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
alpha_contours, alpha_hierarchy = cv2.findContours(alpha_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cv2.imshow('Target', target_img)

for c in target_contours:
    x,y,w,h = cv2.boundingRect(c)
    curt = target_img[y:y+h, x:x+w]
    print_image_wait('Curt', curt)


'''
list_char = list()
for c in target_contours:
    score = 10.0
    global cour
    isEnter = False
    for l, i in zip(alpha_contours, range(len(alpha_contours))):
        ret = cv2.matchShapes(c, l, cv2.CONTOURS_MATCH_I3, 0.0)

        if ret < score:
            score = ret
            cour = i
            isEnter = True
    if isEnter == True:
        isEnter = False
        list_char.append(cour)

print(list_char)


alphabet = list()

alphabet.append([alpha_contours[25], 'A'])
alphabet.append([alpha_contours[26], 'B'])
alphabet.append([alpha_contours[24], 'C'])
alphabet.append([alpha_contours[23], 'D'])
alphabet.append([alpha_contours[22], 'E'])
alphabet.append([alpha_contours[21], 'F'])
alphabet.append([alpha_contours[20], 'G'])
alphabet.append([alpha_contours[19], 'H'])
alphabet.append([alpha_contours[18], 'I'])
alphabet.append([alpha_contours[17], 'J'])
alphabet.append([alpha_contours[16], 'K'])
alphabet.append([alpha_contours[15], 'L'])
alphabet.append([alpha_contours[14], 'M'])
alphabet.append([alpha_contours[13], 'N'])
alphabet.append([alpha_contours[12], 'Q'])
alphabet.append([alpha_contours[11], 'O'])
alphabet.append([alpha_contours[10], 'P'])
alphabet.append([alpha_contours[9], 'R'])
alphabet.append([alpha_contours[8], 'S'])
alphabet.append([alpha_contours[7], 'T'])
alphabet.append([alpha_contours[6], 'U'])
alphabet.append([alpha_contours[5], 'Y'])
alphabet.append([alpha_contours[4], 'V'])
alphabet.append([alpha_contours[3], 'W'])
alphabet.append([alpha_contours[2], 'X'])
alphabet.append([alpha_contours[1], 'Z'])
'''
