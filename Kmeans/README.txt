Open the file "kMeans_config.txt"

Please enter 4 lines as follows:

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
    Enter the maximum number of iterations for the kMeans algorithm
    Eg: 30

Sample kMeans_config lines:
hard
3,5,9
new_dataset_1.txt
30
