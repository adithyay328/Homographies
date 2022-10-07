"""This script drives a homography computation system. This component takes in 2
images, visualizes all detected edges, and then calls into a RANSAC module that then
"""

import os
import sys
import numpy as np
import cv2 as cv

import RANSACHomography

# Loading filenames
fNameOne = sys.argv[-2]
fNameTwo = sys.argv[-1]

print(fNameOne)
print(fNameTwo)

# Loading images
imgOne = cv.imread(fNameOne)
imgTwo = cv.imread(fNameTwo)

# We need to downscale them to be able to view them on screen when displayed
downScale = lambda img, scale: cv.resize(img, ( int(img.shape[1] * scale) , int(img.shape[0] * scale)  ), 
  interpolation = cv.INTER_AREA  )

newScale = 0.225
imgOne = downScale(imgOne, newScale)
imgTwo = downScale(imgTwo, newScale)

# A function to compute good points to track
def getGoodPoints(img):
  goodFeaturesClone = img.copy()
  gray = cv.cvtColor(goodFeaturesClone,cv.COLOR_BGR2GRAY)
  
  # There are some feature quality params in here - change
  # if you want 
  goodFeatures = cv.goodFeaturesToTrack(gray, 300, 0.05, 20)
  
  # Remove the outer array from each point
  goodFeatures = [goodFeature[0] for goodFeature in goodFeatures]

  return goodFeatures

def displayGoodPoints(img, name, goodFeatures):
  goodFeaturesClone = img.copy()
  for point in goodFeatures:
    cv.circle(goodFeaturesClone, (int(point[0]), int(point[1])), 2, color=(255, 0, 0), thickness=5)
    
  cv.imshow(f"GoodFeatures{name}", goodFeaturesClone)

imgOneGoodPoints = getGoodPoints(imgOne)
imgTwoGoodPoints = getGoodPoints(imgTwo)

# Convert to keypoints for SIFT
imgOneKeypoints = [cv.KeyPoint(goodPoint[0], goodPoint[1], 1) for goodPoint in imgOneGoodPoints]
imgTwoKeypoints = [cv.KeyPoint(goodPoint[0], goodPoint[1], 1) for goodPoint in imgTwoGoodPoints]

# Compute SIFT descriptors
# orb = cv.ORB_create()
sift = cv.xfeatures2d.SIFT_create()
_, imgOneDes = sift.compute(imgOne, imgOneKeypoints)
_, imgTwoDes = sift.compute(imgTwo, imgTwoKeypoints)

## Get matches
# bf = cv.BFMatcher(crossCheck=True)
# matches = bf.match(imgOneDes, imgTwoDes)

# Get matches with ratio test
bf = cv.BFMatcher()
rawMatches = bf.knnMatch(imgOneDes, imgTwoDes, k=2)

# Good matches
matches = []
for m,n in rawMatches:
  # Minumum distance difference
  MIN_DELTA = 0.25

  if m.distance < (1 - MIN_DELTA) * n.distance:
    matches.append(m)

matImg = cv.drawMatches(imgOne, imgOneKeypoints, imgTwo, imgTwoKeypoints, matches, None)

# Just give a visual so that the user can understand what the
# points look like
# displayGoodPoints(imgOne, "one", imgOneGoodPoints)
# displayGoodPoints(imgTwo, "two", imgTwoGoodPoints)
cv.imshow("Matches", matImg)

k = cv.waitKey(0)

# Combine the keypoint coords and descriptors into one list
# per image; just to pass into our RANSAC matcher
firstImageTups = [ ( *imgOneGoodPoints[i] , imgOneDes[i] ) for i in range(len(imgOneDes)) ]
secondImageTups = [ ( *imgTwoGoodPoints[i] , imgTwoDes[i] ) for i in range(len(imgTwoDes)) ]

# Compute homography matrix
h = RANSACHomography.computeHomography(firstImageTups, secondImageTups, matches)

# Now, time to visualize. Show the first image with the original keypoints, and the
# output with predicted position
displayGoodPoints(imgOne, "one", imgOneGoodPoints)

predictedPointsSecond = []
# Now, compute points on the second image
for point in imgOneGoodPoints:
  firstPointHomog = np.array([point[0], point[1], 1])
  # Predicted homogenous
  predictedHomog = np.matmul(h, firstPointHomog)
  # Casting homogenous back to 2d
  predicted2D = np.array([predictedHomog[0], predictedHomog[1]]) / predictedHomog[-1]

  predictedPointsSecond.append(predicted2D)

displayGoodPoints(imgTwo, "twoPredict", predictedPointsSecond)

# Show projective warping
img1_warp = cv.warpPerspective(imgOne, h, imgOne.shape[:2])

# cv.imshow("Warped", img1_warp)

k = cv.waitKey(0)