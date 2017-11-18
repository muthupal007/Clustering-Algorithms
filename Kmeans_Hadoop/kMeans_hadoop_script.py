import numpy as np
import random
import os
import subprocess
import tempfile
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


'''-------------get k initial centroids-------------'''


def get_initial_centroids(k_value):
    # initialize the centroids with k random points
    rand_num = random.sample(range(0, adjusted_matrix.shape[0]), k=k_value)
    centroids = adjusted_matrix[rand_num, :]

    return centroids


'''-------------write mapReduce input file-------------'''


def prepare_map_reduce_input_file(centroids, feature_matrix):
    # concatenate the centroids with the feature matrix and write to mapReduce input file
    centroids_features_matrix = np.concatenate((centroids, feature_matrix))
    np.savetxt(map_reduce_input_file_name, centroids_features_matrix, delimiter=' ')


'''-------------get mapReduce command-------------'''


def get_map_reduce_command(hadoop_streamer_path_local, hadoop_fs_input_file_local, hadoop_fs_output_folder_local, k_val):
    k_string = str(k_val)
    job_count = str(k_val)
    num_mappers = job_count
    num_reducers = job_count

    return "hadoop jar " + hadoop_streamer_path_local + " -Dmapreduce.job.maps=" + num_mappers + " -Dmapreduce.job.reduces=" + num_reducers + " -files ./mapper.py,./reducer.py -mapper ./mapper.py  -reducer ./reducer.py -cmdenv K_VALUE=" + k_string + " -input " + hadoop_fs_input_file_local + " -output " + hadoop_fs_output_folder_local


'''-------------prepare hadoop file system-------------'''


def prepare_hadoop_file_system(hadoop_fs_input_folder_local, map_reduce_input_file_name_local, hadoop_fs_output_folder_local):
    # delete existing mapReduce input file in hadoop file system
    os.system("hadoop fs -rm " + hadoop_fs_input_folder_local + map_reduce_input_file_name_local)

    # copy the new mapReduce input file to hadoop file system
    os.system("hadoop fs -put ./" + map_reduce_input_file_name_local + " " + hadoop_fs_input_folder_local)

    # delete existing mapReduce output folder in hadoop file system
    os.system("hadoop fs -rm -r " + hadoop_fs_output_folder_local)


'''-------------get cluster dictionary-------------'''


def get_cluster_dict(hadoop_fs_output_folder_local):
    cluster_dict_local = dict()

    with tempfile.TemporaryFile() as tempf:
        proc = subprocess.Popen(["hadoop", "fs", "-cat", hadoop_fs_output_folder_local + "/part*"], stdout=tempf)
        proc.wait()
        tempf.seek(0)
        lines = tempf.readlines()

        for line in lines:
            refined_line = line.decode('ascii').strip()
            centroid_index_string, points_indices_string, new_centroid_string = refined_line.split("\t")
            new_centroid = np.fromstring(new_centroid_string, dtype='float', sep=',')
            centroid_index = int(centroid_index_string)
            points_indices_list = [int(point_index_string) for point_index_string in points_indices_string.split(",")]
            cluster_dict_local[centroid_index] = [points_indices_list, new_centroid]

    return cluster_dict_local


'''-------------mapReduce kMeans-------------'''


def map_reduce_k_means(centroids, feature_matrix, hadoop_fs_input_folder_local, map_reduce_input_file_name_local, hadoop_fs_output_folder_local, map_reduce_command_local):

    centroids_old = np.zeros(centroids.shape)
    iterations = 0
    while not np.array_equal(centroids, centroids_old):
        iterations += 1
        # write new mapReduce input file
        prepare_map_reduce_input_file(centroids, feature_matrix)

        # prepare hadoop file system for mapReduce iteration
        prepare_hadoop_file_system(hadoop_fs_input_folder_local, map_reduce_input_file_name_local, hadoop_fs_output_folder_local)

        # perform mapReduce to generate point_indices closest to each centroid
        os.system(map_reduce_command_local)

        # read output of map_reduce into a cluster dictionary with
        # centroids as keys and point indices and cluster_mean as values
        cluster_dict = get_cluster_dict(hadoop_fs_output_folder_local)

        # assign current centroids to old centroids
        np.copyto(centroids_old, centroids)

        # find new centroids of the obtained clusters
        for centroid_index in cluster_dict:
            centroids[centroid_index, :] = cluster_dict[centroid_index][1]

    print("total iterations: " + str(iterations))
    # assign labels for the final clusters
    label_list = np.ones(feature_matrix.shape[0]) * -1

    for point_index in range (feature_matrix.shape[0]):
        for centroid_index in cluster_dict:
            if point_index in cluster_dict[centroid_index][0]:
                label_list[point_index] = centroid_index
    return label_list


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



# declare config parameters for hadoop
with open('hadoop_config.txt') as f:
    lines = f.readlines()

# check if the run is for random indices or hard-coded indices
random_run = False

if lines[0].strip() == "random":
    random_run = True

map_reduce_input_file_name = lines[6].strip()
hadoop_fs_input_folder = lines[4].strip()
hadoop_streamer_path = lines[3].strip()
hadoop_fs_input_file = hadoop_fs_input_folder + map_reduce_input_file_name
hadoop_fs_output_folder = hadoop_fs_input_folder + lines[5].strip()

# reading input
input_file = lines[2].strip()
data = np.loadtxt(input_file, dtype='float')
data_feature_matrix = data[:, 2:]

# pre-processing
# adjusted_matrix = pre_process(data_feature_matrix)
adjusted_matrix = data_feature_matrix


k = 0
initial_centroids = []

# if random run
if random_run:
    # get k initial centroids
    k = 5
    initial_centroids = get_initial_centroids(k)

# if hard-coded run
else:
    # get k initial centroids
    initial_centroid_indices = [int(centroid_index) - 1 for centroid_index in lines[1].strip().split(",")]
    initial_centroids = adjusted_matrix[initial_centroid_indices,:]
    k = len(initial_centroid_indices)

# get mapReduce command
map_reduce_command = get_map_reduce_command(hadoop_streamer_path, hadoop_fs_input_file, hadoop_fs_output_folder, k)

# perform mapReduce k-means
cluster_id_list = map_reduce_k_means(initial_centroids, adjusted_matrix, hadoop_fs_input_folder, map_reduce_input_file_name, hadoop_fs_output_folder, map_reduce_command)
# PCA
plot_pca(cluster_id_list, adjusted_matrix)

# Jaccard
groundTruth_cluster_id_list = data[:, 1]
jaccard_similarity = get_jaccard_similarity(adjusted_matrix, cluster_id_list, groundTruth_cluster_id_list)
print("Jaccard similarity: " + str(jaccard_similarity))