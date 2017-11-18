NOTE: Ensure that "kMeans_hadoop_script.py", "mapper.py" and "reducer.py"

Open the file "hadoop_config.txt"

Please enter 7 lines as follows:

Line 1:
    Enter "hard" for hard-coding the gene_ids
    Enter "random" for auto-assigning random indices

Line 2: (can be an empty line if Line 1 is set to "random")
    Enter the gene_ids to hard-code separated by comma. K is determined based on the number of gene_ids mentioned in the list.
    Eg: "3,5,9"

Line 3:
    Enter the full-path and name of input file
    Eg: "new_dataset_1.txt"

Line 4:
    Enter full path of hadoop streaming jar file
    Eg: "/usr/local/Cellar/hadoop/2.8.1/libexec/share/hadoop/tools/lib/hadoop-streaming-2.8.1.jar"

Line 5:
    Enter path of work folder in hadoop file system. This would be the location of the input file and output directory
    Eg: "/"

Line 6:
    Enter name of the output folder to be created in hadoop file system
    Eg: "kMeans_output_folder"

Line 7:
    Enter name of input file to be created and placed in the hadoop file system
    Eg: "centroids_and_features.txt"



Sample hadoop_config lines:
random
3,5,9
new_dataset_1.txt
/usr/local/Cellar/hadoop/2.8.1/libexec/share/hadoop/tools/lib/hadoop-streaming-2.8.1.jar
/
kMeans_output_folder
centroids_and_features.txt
