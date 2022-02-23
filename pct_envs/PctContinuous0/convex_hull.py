#--coding:utf-8--
import numpy as np

class Line2D(object):
    # constructor for the line class
    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2
        # calculate the slope of the line
        if self.p2[0] != self.p1[0]:
            self.slope = (self.p2[1] - self.p1[1]) \
                     / (self.p2[0] - self.p1[0])
        else:
            self.slope = (self.p2[1] - self.p1[1]) * np.inf

    def orientation(self,line2):
        # get slopes
        slope1 = self.slope
        slope2 = line2.slope

        if abs(slope1) == np.inf and abs(slope2) == np.inf:
            return 0

        # get difference
        diff = slope2-slope1
        # determine orientation
        if diff > 0:
            return -1 # counter-clockwise
        elif diff == 0:
            return 0  # colinear
        else:
            return 1  # clockwise

def sortPoints(point_list):
    point_list = sorted(point_list,key = lambda x:x[0])
    # This function do not account for comparison of two points at the same x-location. Resolve!
    return point_list

def ConvexHull(point_list):
    # initalize two empty lists for upper
    # and lower hulls.
    point_list = np.array(point_list).astype(np.float)
    point_list[:,0] += point_list[:,1] * 1e-6
    point_list = point_list.tolist()
    upperHull = []
    lowerHull = []
    # sort the list of 2D-points
    sorted_list = sortPoints(point_list)

    for point in sorted_list:
        if len(lowerHull) >= 2:
            line1 = Line2D(lowerHull[len(lowerHull) - 2],
                           lowerHull[len(lowerHull) - 1])
            line2 = Line2D(lowerHull[len(lowerHull) - 1],
                           point)
        while len(lowerHull) >= 2 and line1.orientation(line2) != -1:
            removed = lowerHull.pop()
            if lowerHull[0] == lowerHull[len(lowerHull) - 1]:
                break
            # set the last two lines in lowerHull
            line1 = Line2D(lowerHull[len(lowerHull) - 2],
                           lowerHull[len(lowerHull) - 1])
            line2 = Line2D(lowerHull[len(lowerHull) - 1],
                           point)
        lowerHull.append(point)
        # reverse the list for upperHull search
    reverse_list = sorted_list[::-1]
    for point in reverse_list:
        if len(upperHull) >= 2:
            line1 = Line2D(upperHull[len(upperHull) - 2],
                           upperHull[len(upperHull) - 1])
            line2 = Line2D(upperHull[len(upperHull) - 1],
                           point)
        while len(upperHull) >= 2 and \
                line1.orientation(line2) != -1:
            removed = upperHull.pop()
            if upperHull[0] == upperHull[len(upperHull) - 1]:
                break
            # set the last two lines in lowerHull
            line1 = Line2D(upperHull[len(upperHull) - 2],
                           upperHull[len(upperHull) - 1])
            line2 = Line2D(upperHull[len(upperHull) - 1],
                           point)
        upperHull.append(point)

    # final touch: remove the last members
    # of each point as they are the same as
    # the first point of the complementary set.
    removed = upperHull.pop()
    removed = lowerHull.pop()
    # concatenate lists
    convexHullPoints = lowerHull + upperHull
    convexHullPoints = np.array(convexHullPoints)

    return convexHullPoints

def point_in_polygen(point, coords):
    lat, lon = point
    polysides = len(coords)
    j = polysides - 1
    oddnodes = False

    for i in range(polysides):
        if np.sum(np.cross(coords[i] - point, point - coords[j])) == 0:
            return False

        if (coords[i][1] < lon and coords[j][1] >= lon) or (coords[j][1] < lon and coords[i][1] >= lon):
            if (coords[i][0] + (lon - coords[i][1]) / (coords[j][1] - coords[i][1]) * (coords[j][0] - coords[i][0])) < lat:
                oddnodes = not oddnodes
        j = i

    return oddnodes


