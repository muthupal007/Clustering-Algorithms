#!/usr/bin/env python

import sys
import numpy as np
import os

input_matrix = np.loadtxt(sys.stdin, dtype='float')
k_value = int(os.environ['K_VALUE'])
# k_value = 5

centroids_matrix = input_matrix[0:k_value, :]
feature_matrix = input_matrix[k_value:input_matrix.shape[0], :]


for i in range(feature_matrix.shape[0]):
    point = feature_matrix[i]
    point_string = ','.join(str(feature) for feature in point)
    closest_centroid_index = np.argmin(np.linalg.norm(centroids_matrix - point, axis=1))
    print('%s\t%s\t%s' % (closest_centroid_index, i, point_string))
