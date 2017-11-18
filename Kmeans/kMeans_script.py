import numpy as np
import random
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

'''-------------pre-processing-------------'''


def pre_process(feature_matrix):
    # feature selection
    redundant_feature_indices = np.argwhere((np.std(feature_matrix, axis=0)) == 0)
    feature_matrix = np.delete(feature_matrix, redundant_feature_indices, axis=1)

    # normalize the features
    preprocessed_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (
    np.max(feature_matrix, axis=0) - np.min(feature_matrix, axis=0))
    return preprocessed_matrix

'''-------------k-means-------------'''


def get_k_clusters(k_value, break_count, centroids):
    # initialize all labels to -1
    label_list = np.ones(adjusted_matrix.shape[0]) * -1

    # initialize old centroids to a zero matrix
    centroids_old = np.ones(centroids.shape) * -1

    centroids_points_indices_dict = dict()

    # perform k means until convergence
    iterations = 0
    while (not np.array_equal(centroids, centroids_old)) and (break_count > 0):
        iterations += 1
        break_count -= 1
        for b in range(0, adjusted_matrix.shape[0]):
            label_list[b] = np.argmin(np.linalg.norm(centroids - adjusted_matrix[b], axis=1))
        np.copyto(centroids_old, centroids)
        for x in range(0, k_value):
            centroids[x, :] = np.mean(adjusted_matrix[np.where(label_list == x)], axis=0)
            centroids_points_indices_dict[x] = np.asarray(np.where(label_list == x)) + 1

    print("total iterations: " + str(iterations))
    for (key, val) in centroids_points_indices_dict.items():
        print(str(key) + ": " + str(val.flatten()))

    centroid_points_by_id = np.zeros(centroids.shape)
    for i in range(centroids.shape[0]):
        centroid_points_by_id[i] = [(feature + 1) for feature in centroids[i]]


    # calculate the SSE
    sse_local = 0

    for l in range(0, k_value):
        sse_local += np.sum(np.linalg.norm(adjusted_matrix[np.where(label_list == l)] - centroids[l, :], axis=1) ** 2)

    return [sse_local, label_list]


def k_means(k_value, break_count, m, centroids):
    # perform k_means and obtain the minimum sse
    sse_min = sys.float_info.max
    for z in range(0, m):
        sse_local, label_list = get_k_clusters(k_value, break_count, centroids)
        if sse_local < sse_min:
            sse_min = sse_local

    return [sse_min, label_list]


'''-------------PCA-------------'''


def plot_pca(classes_list, feature_matrix):
    # get the unique list of classes
    unique_classes_list = list(set(classes_list))

    # obtain the principle components matrix
    pca_object = PCA(n_components=2, svd_solver='full')
    pca_object.fit(feature_matrix)
    principle_components_matrix = pca_object.transform(feature_matrix)

    # plot cluster_ids using the principle components as the coordinates and classes as labels
    colors = [plt.cm.jet(float(i) / max(unique_classes_list)) for i in unique_classes_list]
    for i, u in enumerate(unique_classes_list):
        xi = [p for (j,p) in enumerate(principle_components_matrix[:,0]) if classes_list[j] == u]
        yi = [p for (j,p) in enumerate(principle_components_matrix[:,1]) if classes_list[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(int(u)))

    plt.title(input_file.split(".")[0] + " scatter plot")
    plt.xlabel("Principle_component_1")
    plt.ylabel("Principle_component_2")
    plt.legend()
    plt.show()


'''-------------Jaccard-------------'''


def get_jaccard_similarity(clustered_feature_matrix, classes_list, ground_truth_classes_list):
    obtained_same_cluster_matrix = np.zeros((len(clustered_feature_matrix), len(clustered_feature_matrix)))
    ground_truth_same_cluster_matrix = np.zeros((len(clustered_feature_matrix), len(clustered_feature_matrix)))

    # populate the same cluster matrices
    for i in range(obtained_same_cluster_matrix.shape[0]):
        obtained_same_cluster_matrix[i][i] = 1
        ground_truth_same_cluster_matrix[i][i] = 1
        for j in range(i + 1, obtained_same_cluster_matrix.shape[1]):
            if classes_list[i] == classes_list[j]:
                obtained_same_cluster_matrix[i][j] = 1
                obtained_same_cluster_matrix[j][i] = 1
            if ground_truth_classes_list[i] == ground_truth_classes_list[j]:
                ground_truth_same_cluster_matrix[i][j] = 1
                ground_truth_same_cluster_matrix[j][i] = 1

    # calculate the jaccard similarity
    numerator = np.sum(np.logical_and(obtained_same_cluster_matrix, ground_truth_same_cluster_matrix))
    denominator = np.sum(np.logical_or(obtained_same_cluster_matrix, ground_truth_same_cluster_matrix))
    return numerator / denominator


'''-------------SSE-------------'''


def plot_sse(iter_local, min, max):
    # store the sse obtained for each cluster_count in a list
    sse_list = np.zeros(max)
    for cluster_count in range(min, max + 1):
        rand_num = random.sample(range(0, adjusted_matrix.shape[0]), k=cluster_count)
        centroids = adjusted_matrix[rand_num, :]
        sse_local, cluster_id_list_local = k_means(cluster_count,break_count, iter_local, centroids)
        sse_list[cluster_count - 1] = sse_local

    # plot the SSE graph
    plt.plot(sse_list)
    plt.ylabel('SSE')
    plt.xlabel('k')
    plt.show()


# reading input
with open('kMeans_config.txt') as f:
    lines = f.readlines()

# check if the run is for random indices or hard-coded indices
random_run = False

if lines[0].strip() == "random":
    random_run = True

input_file = lines[2].strip()
data = np.loadtxt(input_file, dtype='float')
data_feature_matrix = data[:, 2:]

# pre-processing
# adjusted_matrix = pre_process(data_feature_matrix)
adjusted_matrix = data_feature_matrix

# read input break_count
break_count = int(lines[3].strip())

initial_centroid_indices = []
k = 0
centroids = []
# if hard-coded run
if not random_run:
    # initialize the centroids with file input, index is id - 1
    initial_centroid_indices = [int(centroid_index) - 1 for centroid_index in lines[1].strip().split(",")]
    k = len(initial_centroid_indices)
    centroids = adjusted_matrix[initial_centroid_indices, :]

# if random run
else:
    # initialize the centroids with random points
    k=5
    initial_centroid_indices = random.sample(range(0, adjusted_matrix.shape[0]), k=k)
    centroids = adjusted_matrix[initial_centroid_indices, :]

    # SSE Plot
    iter = 10
    min_num_of_clusters = 1
    max_num_of_clusters = 15
    plot_sse(iter, min_num_of_clusters, max_num_of_clusters)

# k-means
sse, cluster_id_list = k_means(k, break_count, 1, centroids)

# PCA
plot_pca(cluster_id_list, adjusted_matrix)

# Jaccard
groundTruth_cluster_id_list = data[:, 1]
jaccard_similarity = get_jaccard_similarity(adjusted_matrix, cluster_id_list, groundTruth_cluster_id_list)
print("Jaccard similarity: " + str(jaccard_similarity))