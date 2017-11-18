import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def dist_mat(myList):

    l = myList.shape[0]
    distance_matrix = np.zeros((int((l**2-l)/2), 3))
    counter = 0
    for x in range(0,(myList.shape[0])-1):
        for y in range((x+1),(myList.shape[0])):
            distance_matrix[counter] = [x, y, (np.linalg.norm(myList[x]-myList[y],axis=0))]
            counter += 1

    return(distance_matrix)

def hc(inp,k):
    distance_matrix = dist_mat(inp)
    distance_matrix = pd.DataFrame(distance_matrix)
    distance_matrix = distance_matrix.sort_values(by=2)
    distance_matrix.reset_index(inplace=True)
    distance_matrix = distance_matrix.iloc[:,1:4]
    count = inp.shape[0]
    label = np.array(list(range(0,inp.shape[0])))
    for l in range(0,distance_matrix.shape[0]):
        if(count!=k):
            if((distance_matrix.loc[l][0])!=(distance_matrix.loc[l][1])):
                label[int(distance_matrix.loc[l][0])] = distance_matrix.loc[l][0]
                label[label==distance_matrix.loc[l][1]] = distance_matrix.loc[l][0]
                distance_matrix = distance_matrix.replace((distance_matrix.loc[l][1]),(distance_matrix.loc[l][0]))
                count = count-1


    unique, counts = np.unique(label, return_counts= True)
    unique = [int(i) for i in unique]
    print("Cluster ID: count = " + str(dict(zip(unique, counts))))

    centroids = {}
    for x in label:
        if x == 0:
            continue
        centroids[x] = np.asarray(np.where(label == x))+1

    #Printing Centroids
    print("Cluster ID: points in cluster = ")
    for key, value in centroids.items():
        print (str(key) + ":" + str(value))

    return label


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
            if classes_list[i] != -1 and classes_list[j] != -1 and classes_list[i] == classes_list[j]:
                obtained_same_cluster_matrix[i][j] = 1
                obtained_same_cluster_matrix[j][i] = 1
            if ground_truth_classes_list[i] != -1 and ground_truth_classes_list[j] != -1 and ground_truth_classes_list[i] == ground_truth_classes_list[j]:
                ground_truth_same_cluster_matrix[i][j] = 1
                ground_truth_same_cluster_matrix[j][i] = 1

    # calculate the jaccard similarity
    numerator = np.sum(np.logical_and(obtained_same_cluster_matrix, ground_truth_same_cluster_matrix))
    denominator = np.sum(np.logical_or(obtained_same_cluster_matrix, ground_truth_same_cluster_matrix))
    return numerator / denominator

input_file = input('Enter input data: ')
k = int(input('Enter no. of clusters required: '))
data = np.loadtxt(input_file,dtype='float')
y = data[:,2:]

# Removing columns with 0 variance/std
ab = np.argwhere((np.std(y,axis=0))==0)
y = np.delete(y, ab, axis = 1)

# normalization
adjusted_matrix = (y - y.mean(axis=0))/(np.max(y,axis=0)-np.min(y,axis=0))

cluster_id_list = hc(adjusted_matrix,k)

# PCA
unique_cluster_id_list = list(set(cluster_id_list))
new_list = []
for cluster_id in cluster_id_list:
    new_list.append(unique_cluster_id_list.index(cluster_id))

plot_pca(new_list, adjusted_matrix)

# Jaccard
groundTruth_cluster_id_list = data[:, 1]
jaccard_similarity = get_jaccard_similarity(adjusted_matrix, cluster_id_list, groundTruth_cluster_id_list)
print("Jaccard similarity: " + str(jaccard_similarity))

