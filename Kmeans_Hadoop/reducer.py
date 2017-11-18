#!/usr/bin/env python
import sys
import numpy as np

current_centroid = None
current_points = ""
centroid = None
count_of_points = 0
sum_array = None

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    centroid, point_index, point_string = line.split('\t')
    point = np.fromstring(point_string, dtype='float', sep=',')

    # this IF-switch only works because Hadoop sorts map output by key before it is passed to the reducer
    if current_centroid == centroid:
        current_points += "," + point_index
        count_of_points += 1
        sum_array += point
    else:
        if current_centroid:
            # write result to STDOUT
            new_centroid = sum_array / count_of_points
            new_centroid_string = ','.join(str(feature) for feature in new_centroid)
            print ('%s\t%s\t%s' % (current_centroid, current_points, new_centroid_string))
        current_points = point_index
        current_centroid = centroid
        count_of_points = 1
        sum_array = np.copy(point)

# do not forget to output the last word if needed!
if current_centroid == centroid:
    new_centroid = sum_array / count_of_points
    new_centroid_string = ','.join(str(feature) for feature in new_centroid)
    print ('%s\t%s\t%s' % (current_centroid, current_points, new_centroid_string))
