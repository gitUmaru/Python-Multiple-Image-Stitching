import cv2
import numpy as np
import glob
import argparse
import imutils

def normalize_image(imageFile):
    # load image as grayscale
    img = cv2.imread(imageFile)

    gray = cv2.imread(imageFile,0)

    # threshold input image using otsu thresholding as mask and refine with morphology
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # put mask into alpha channel of image
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    # save resulting masked image
    cv2.imwrite('images\\'+imageFile[18:26]+'.png', result)
    print(imageFile[18:26]+'.png' + " has been normed")

    # display result, though it won't show transparency
    # cv2.imshow("RESULT", result)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    filenames = glob.glob('images/*.png')
    print(filenames)

    for file in filenames:
        print(file)

    print("ALL IMAGES HAVE BEEN NORMALIZED")

    img = cv2.imread("images\jpg_images\IMG_3793.jpg")
    print(img)
    crop_img = img[1000:3000, 1000:3000]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
