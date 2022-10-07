"""Utilizes RANSAC to compute the most accurate 3x3 homography matrix
between 2 sets of points and descriptors."""

import random

import numpy as np

# The distance metric we're using is hamming distance, which is efficient
# for binary descriptors stored as vectors
def hammingDist(binOne, binTwo):
  assert len(binOne) == len(binTwo)
  dist = 0
  for i in range(len(binOne)):
    if binOne[i] != binTwo[i]:
      dist += 1
  
  return dist
  
"""Computes homography by repeatedly running RANSAC in order to approximate
the True homography matrix betwen 2 sets of points and descriptors."""
def computeHomography(imgOneTups, imgTwoTups, matches, maxIterations = 100):
  # A list of all previously computed solution matrices. We pick the one
  # with the highest agreement as the final solution
  resultantMatrices = []

  for iteration in range(maxIterations):
    # Pick tuples based on some random subset of the matches
    numMatches = 4
    selectedDMatches = random.sample(matches, numMatches)
    
    matchingTuples = [(imgOneTups[dMatch.queryIdx], imgTwoTups[dMatch.trainIdx]) for dMatch in selectedDMatches]

    # At this point, we have a list of all matching tuples. So now, apply
    # the direct lienar transform to each pair, and then throw into the following list.
    # We'll solve the resulting homogenous linear system later
    Arows = []

    for i in range(len(matchingTuples)):
      currImgOneTuple = matchingTuples[i][0]
      currImgTwoTuple = matchingTuples[i][1]

      # Extracting needed coordinates
      x = currImgOneTuple[0]
      y = currImgOneTuple[1]
      xP = currImgTwoTuple[0]
      yP = currImgTwoTuple[1]

      # Add the first and second rows, then append each Arows. These needs to be
      # v-stacked
      rowOne = np.array([-x, -y, -1, 0, 0, 0, x*xP, y*xP, xP])
      rowTwo = np.array([0, 0, 0, -x, -y, -1, x*yP, y*yP, yP])

      # Add to list
      Arows.append(rowOne)
      Arows.append(rowTwo)
    
    # Solve for homography using SVD; basically, pick the row of I
    # that corresponds to either a 0, or the smallest singular value
    # otherwise. Prefer the 0 case
    A = np.vstack(Arows)

    u, s, v = np.linalg.svd(A)
    
    # We need to find the row that corresponds to a 0, or the smallest
    # non-singular if you can't get a 0
    solutionIndex = -1
    # if len(np.argwhere(s == 0)) != 0:
    #   solutionIndex = np.argwhere(s == 0)[0][0]
    # else:
    #   solutionIndex = np.argmin(s)
    solution = v[solutionIndex].reshape( (3, 3) )
    
    # Now, we need to apply RANSAC to identify if this is a good
    # homography or not. Stage 1 fits the matrix to the subset,
    # then computes if this homography makes sense on the whole dataset subset by seeing
    # how many points are within some threshold epsilon, and terminates if the percentage
    # of inliers is below some threshold rho.
    # If you pass stage 1, stage 2 involves refitting the model to all inliers.

    # Then, some global loss function is used to pick the best homography

    # The maximum distance between prediction and outcome
    epsilon = 5
    # Proportion of points that need to fit
    rho = 0.1

    # Stage 1: compute global number of inliers
    inliers = []
    for match in matches:
      img1Idx = match.queryIdx
      img2Idx = match.trainIdx

      img1Point = np.array([*imgOneTups[img1Idx][0:2]])
      img2Point = np.array([*imgTwoTups[img2Idx][0:2]])

      # Compute predicted point
      img1Homog = np.array([*img1Point, 1])
      predicedHomog = np.matmul(solution, img1Homog)
      predicted2d = predicedHomog[0:2] / predicedHomog[-1]

      dist = np.linalg.norm(img2Point - predicted2d)

      # Add to inlier list if it's a good match
      if dist <= epsilon:
        inliers.append(match)
    
    # Stage 2: if the number of inliers is greater than rho, refit on all inliers and
    # add to resultantMatrices
    if (len(inliers) / len(matches)) < rho:
      continue
    
    # Fit to all inliers
    matchingTuples = [(imgOneTups[dMatch.queryIdx], imgTwoTups[dMatch.trainIdx]) for dMatch in inliers]
    Arows = []

    for i in range(len(matchingTuples)):
      currImgOneTuple = matchingTuples[i][0]
      currImgTwoTuple = matchingTuples[i][1]

      # Extracting needed coordinates
      x = currImgOneTuple[0]
      y = currImgOneTuple[1]
      xP = currImgTwoTuple[0]
      yP = currImgTwoTuple[1]

      # Add the first and second rows, then append each Arows. These needs to be
      # v-stacked
      rowOne = np.array([-x, -y, -1, 0, 0, 0, x*xP, y*xP, xP])
      rowTwo = np.array([0, 0, 0, -x, -y, -1, x*yP, y*yP, yP])

      # Add to list
      Arows.append(rowOne)
      Arows.append(rowTwo)
    
    # Solve for homography using SVD; basically, pick the row of I
    # that corresponds to either a 0, or the smallest singular value
    # otherwise. Prefer the 0 case
    A = np.vstack(Arows)

    u, s, v = np.linalg.svd(A)
    
    # We need to find the row that corresponds to a 0, or the smallest
    # non-singular if you can't get a 0
    solutionIndex = -1
    
    goodSol = v[solutionIndex].reshape( (3, 3) )
    resultantMatrices.append(goodSol)

  h = random.choice(resultantMatrices)
  print(h)
  return h