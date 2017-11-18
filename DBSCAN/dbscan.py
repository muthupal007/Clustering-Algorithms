
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


'''-------------pre-processing-------------'''



def dbscan(data_feature_matrix, eps, minpts):

    clusterid = 0
    label = np.zeros(data_feature_matrix.shape[0])
    
    for pt in range(0, data_feature_matrix.shape[0]):

        if label[pt] != 0: #already visited
            continue
        
        neighbors = regionQuery(data_feature_matrix, eps, pt)
        if len(neighbors) < minpts:
            label[pt] = -1 #assign point to noise
            continue
        clusterid += 1
        expandcluster(pt, neighbors, clusterid, eps, minpts, label)


    unique, counts = np.unique(label, return_counts= True)
    unique = [int(i) for i in unique]
    print("Cluster: count = " + str(dict(zip(unique, counts))))

    centroids = {}
    for x in range(int(np.min(label)), int(np.max(label)) + 1):
        if x == 0:
            continue
        centroids[x] = np.asarray(np.where(label == x))+1

    #Printing Centroids
    print("Cluster: points in cluster = ")
    for key, value in centroids.items():
        print (str(key) + ":" + str(value))


    return  label

        
       
        
def expandcluster(pt, neighborpts, clusterid, eps, minpts, label):
    
    label[pt] = clusterid 
    for point in neighborpts:

        if label[point] == -1:
           label[point] = clusterid #border points

        if label[point] == 0:
            label[point] = clusterid
            ptneighbor = regionQuery(data_feature_matrix, eps, point)
            if len(ptneighbor) >= minpts:
                neighborpts += ptneighbor
        
        

def regionQuery(data_feature_matrix, eps, point):
    
    neighbors = []
    
    for otherpt in range(0, data_feature_matrix.shape[0]):
        if np.linalg.norm(data_feature_matrix[point] - data_feature_matrix[otherpt]) <= eps:
            neighbors.append(otherpt)
    
    return neighbors        
                


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
            if ground_truth_classes_list[i] != -1 and ground_truth_classes_list[j] != -1 and ground_truth_classes_list[i] == ground_truth_classes_list[j]:
                ground_truth_same_cluster_matrix[i][j] = 1
                ground_truth_same_cluster_matrix[j][i] = 1

    # calculate the jaccard similarity
    numerator = np.sum(np.logical_and(obtained_same_cluster_matrix, ground_truth_same_cluster_matrix))
    denominator = np.sum(np.logical_or(obtained_same_cluster_matrix, ground_truth_same_cluster_matrix))
    return numerator / denominator

# reading input

input_file = input('Enter input data: ')
data = np.loadtxt(input_file, dtype='float')
data_feature_matrix = data[:, 2:]

epsilon = float(input('Enter epsilon(radius) value: '))
minimumpoints = int(input('Enter the minimum number of points: '))

#DBSCAN
cluster_id_list = dbscan(data_feature_matrix, epsilon, minimumpoints)

plot_pca(cluster_id_list, data_feature_matrix)

#Jaccard
groundTruth_cluster_id_list = data[:, 1]
jaccard_similarity = get_jaccard_similarity(data_feature_matrix, cluster_id_list, groundTruth_cluster_id_list)
print("Jaccard similarity: " + str(jaccard_similarity))
