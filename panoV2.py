'''
Purpose: Program creates a panoramic photo from multiple images
'''
import numpy as np
import cv2
import sys
import os
import math
from PIL import Image
from numpy import linalg


def filterMatches(matches, ratio=0.75):
    filteredMatches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filteredMatches.append(m[0])

    return filteredMatches


def imageDistance(matches):

    sumDistance = 0.0

    for match in matches:

        sumDistance += match.distance

    return sumDistance


def findDimensions(image, homography):
    baseP1 = np.ones(3, np.float32)
    baseP2 = np.ones(3, np.float32)
    baseP3 = np.ones(3, np.float32)
    baseP4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    baseP1[:2] = [0, 0]
    baseP2[:2] = [x, 0]
    baseP3[:2] = [0, y]
    baseP4[:2] = [x, y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [baseP1, baseP2, baseP3, baseP4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hpArr = np.array(hp, np.float32)

        normalPt = np.array(
            [hpArr[0] / hpArr[2], hpArr[1] / hpArr[2]], np.float32)

        if (max_x is None or normalPt[0, 0] > max_x):
            max_x = normalPt[0, 0]

        if (max_y is None or normalPt[1, 0] > max_y):
            max_y = normalPt[1, 0]

        if (min_x is None or normalPt[0, 0] < min_x):
            min_x = normalPt[0, 0]

        if (min_y is None or normalPt[1, 0] < min_y):
            min_y = normalPt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)


def stitchImages(baseImgRGB, directory, output, round, imgType):

    if (len(directory) < 1):
        return baseImgRGB

    baseImg = cv2.GaussianBlur(cv2.cvtColor(
        baseImgRGB, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # Use the SURF feature detector
    detector = cv2.xfeatures2d.SURF_create()

    # Find key points in base image for motion estimation
    baseFeat, baseDescs = detector.detectAndCompute(baseImg, None)

    # Parameters for nearest-neighbor matching
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    flannParams = dict(algorithm=FLANN_INDEX_KDTREE,
                       trees=5)
    matcher = cv2.FlannBasedMatcher(flannParams, {})

    print "Iterating through next images..."

    closestImg = None

    # TODO: Thread this loop since each iteration is independent

    # Find the best next image from the remaining images
    for nxtImgPath in directory:

        print "Reading %s..." % nxtImgPath

        if (keyFrame in nxtImgPath):
            print "\t Skipping %s..." % keyFrameFile
            continue

        # Read in the next image...
        nxtImgRGB = cv2.imread(nxtImgPath)
        nxtImg = cv2.GaussianBlur(cv2.cvtColor(
            nxtImgRGB, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        print "\t Finding points..."

        # Find points in the next frame
        nxtFeat, nxtDescs = detector.detectAndCompute(nxtImg, None)

        matches = matcher.knnMatch(
            nxtDescs, trainDescriptors=baseDescs, k=2)

        print "\t Match Count: ", len(matches)

        matchesSubset = filterMatches(matches)

        print "\t Filtered Match Count: ", len(matchesSubset)

        distance = imageDistance(matchesSubset)

        print "\t Distance from Key Image: ", distance

        averagePointDistance = distance / float(len(matchesSubset))

        print "\t Average Distance: ", averagePointDistance

        kp1 = []
        kp2 = []

        for match in matchesSubset:
            kp1.append(baseFeat[match.trainIdx])
            kp2.append(nxtFeat[match.queryIdx])

        p1 = np.array([k.pt for k in kp1])
        p2 = np.array([k.pt for k in kp2])

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))

        inlierRatio = float(np.sum(status)) / float(len(status))

        if (closestImg is None or inlierRatio > closestImg['inliers']):
            closestImg = {}
            closestImg['h'] = H
            closestImg['inliers'] = inlierRatio
            closestImg['dist'] = averagePointDistance
            closestImg['path'] = nxtImgPath
            closestImg['rgb'] = nxtImgRGB
            closestImg['img'] = nxtImg
            closestImg['feat'] = nxtFeat
            closestImg['desc'] = nxtDescs
            closestImg['match'] = matchesSubset

    print "Closest Image: ", closestImg['path']
    print "Closest Image Ratio: ", closestImg['inliers']

    new_directory = filter(lambda x: x != closestImg['path'], directory)

    H = closestImg['h']
    H = H / H[2, 2]
    H_inv = linalg.inv(H)

    if (closestImg['inliers'] > 0.1):  # and

        (min_x, min_y, max_x, max_y) = findDimensions(
            closestImg['img'], H_inv)

        # Adjust max_x and max_y by base img size
        max_x = max(max_x, baseImg.shape[1])
        max_y = max(max_y, baseImg.shape[0])

        move_h = np.matrix(np.identity(3), np.float32)

        if (min_x < 0):
            move_h[0, 2] += -min_x
            max_x += -min_x

        if (min_y < 0):
            move_h[1, 2] += -min_y
            max_y += -min_y

        print "Homography: \n", H
        print "Inverse Homography: \n", H_inv
        print "Min Points: ", (min_x, min_y)

        mod_inv_h = move_h * H_inv

        imgWidth = int(math.ceil(max_x))
        imgHeight = int(math.ceil(max_y))

        print "New Dimensions: ", (imgWidth, imgHeight)

        # Warp the new image given the homography from the old image
        baseImgWarp = cv2.warpPerspective(
            baseImgRGB, move_h, (imgWidth, imgHeight))
        print "Warped base image"

        nxtImgWarp = cv2.warpPerspective(
            closestImg['rgb'], mod_inv_h, (imgWidth, imgHeight))
        print "Warped next image"

        # Put the base image on an enlarged palette
        enlargedBaseImg = np.zeros((imgHeight, imgWidth, 3), np.uint8)

        print "Enlarged Image Shape: ", enlargedBaseImg.shape
        print "Base Image Shape: ", baseImgRGB.shape
        print "Base Image Warp Shape: ", baseImgWarp.shape

        # Create a mask from the warped image for constructing masked composite
        (ret, dataMap) = cv2.threshold(cv2.cvtColor(
            nxtImgWarp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)

        enlargedBaseImg = cv2.add(enlargedBaseImg, baseImgWarp,
                                  mask=np.bitwise_not(dataMap),
                                  dtype=cv2.CV_8U)

        # Now add the warped image
        finalImg = cv2.add(enlargedBaseImg, nxtImgWarp,
                           dtype=cv2.CV_8U)

        # Crop off the black edges
        darkCrop = cv2.cvtColor(finalImg, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(darkCrop, 1, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
        print "Found %d contours..." % (len(contours))

        maxArea = 0
        bestRec = (0, 0, 0, 0)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # print "Bounding Rectangle: ", (x,y,w,h)

            deltaHeight = h - y
            deltaWidth = w - x

            area = deltaHeight * deltaWidth

            if (area > maxArea and deltaHeight > 0 and deltaWidth > 0):
                maxArea = area
                bestRec = (x, y, w, h)

        if (maxArea > 0):
            print "Maximum Contour: ", maxArea
            print "Best Rectangle: ", bestRec

            finalImgCrop = finalImg[
                bestRec[1]:bestRec[1] + bestRec[3],
                bestRec[0]:bestRec[0] + bestRec[2]]

            finalImg = finalImgCrop

        # Write out the current round
        finalFile = "%s/%d." % (output, round)
        finalFile = finalFile + imgType
        cv2.imwrite(finalFile, finalImg)

        return stitchImages(finalImg, new_directory, output,
                            round + 1, imgType)

    else:

        return stitchImages(baseImgRGB, new_directory, output,
                            round + 1, imgType)


def resizeImages(directory, fileName, width):
    width = int(width)
    for i in range(len(directory)):
        imTemp = Image.open(directory[i])
        wPercent = (width / float(imTemp.size[0]))
        height = int((float(imTemp.size[1]) * float(wPercent)))
        imTemp = imTemp.resize((width, height), Image.BILINEAR)
        imTemp.save(os.path.join(fileName, directory[i]))


# -----------------------------------------------------------------------------
# Inputs
fileName = "../extracted_data"  # file name of where the images are
outputFile = "../results"  # where to output the results of each frame
keyFrameFile = "../extracted_data/frame_4s.jpg"   # key frame (img to start)
width = "800"   # width of stitched img
imgType = "jpg"  # format of stitched img

# Creates a data file to store images
try:
    if not os.path.exists('../results'):
        os.makedirs('../results')
except OSError:
    print('Error: Creating directory of results')

# Breaks down input text to get the key frame name
keyFrame = keyFrameFile.split('/')[-1]

# Open the directory given in the arguments
directory = []
try:
    directory = os.listdir(fileName)
    directory = filter(lambda x: x.find(imgType) > -1, directory)
except OSError:
    print >> sys.stderr, ("Unable to open directory: %s" % fileName)
    sys.exit(-1)

directory = map(lambda x: fileName + "/" + x, directory)
resizeImages(directory, fileName, width)
directory = filter(lambda x: x != keyFrameFile, directory)

baseImgRGB = cv2.imread(keyFrameFile)

finalImg = stitchImages(baseImgRGB, directory, outputFile, 0, imgType)

cv2.destroyAllWindows()
