import json
import numpy as np
from annoy import AnnoyIndex
import os
import subprocess
import sys
import random
from sklearn.neighbors import KDTree

def load_ann_index(filename, vector_size = 150, metric = "angular"):
    """Load an Annoy index from file."""
    print("ANN_ loading ", filename)
    t = AnnoyIndex(vector_size, metric)
    t.load(filename)
    return t


params = {}
#params["intent_embeddings"] = "../full_enc.intent.ann"

#ann_index_intent = load_ann_index(
#                params["intent_embeddings"],
#                vector_size=200,
#            )

params["generic_embeddings"] = "../full_enc.generic.ann"

ann_index_intent = load_ann_index(
                params["generic_embeddings"],
                vector_size=150,
            )


print("n items: ", ann_index_intent.get_n_items())

data = [ann_index_intent.get_item_vector(i) for i in range(ann_index_intent.get_n_items()) if not np.isnan(np.sum(ann_index_intent.get_item_vector(i)))]
_kdTree = KDTree(np.array(data))

with open("../id_to_string_hash.json", "r") as f:
    id_to_hash_dict = json.load(f)

with open("../string_hash_to_id.json", "r") as f:
    hash_to_id_dict = json.load(f)

average, average_latency = 0, 0
average_recall_ann = 0
depth = sys.argv[1]
print("Depth: ", depth)

shuffled_id_dict = list(id_to_hash_dict.items())
random.shuffle(shuffled_id_dict)

default_q = sys.argv[2]
C = int(sys.argv[3])

print("Q: {}, C: {}".format(default_q, C))

default_q = "30"
if sys.argv[4] == "build":
    os.system("rm -r intent-idx-stored")
    print("removed index")
    os.system("target/appassembler/bin/IndexVectors -input ../id_to_embeddings.txt -path intent-idx-stored -encoding fw -stored -fw.q {}".format(sys.argv[2]))

missed = 0
for index, (k, v) in enumerate(shuffled_id_dict):
    if index >= 10000:
        break
    test = subprocess.check_output("target/appassembler/bin/ApproximateNearestNeighborSearch -stored -path intent-idx-stored -encoding fw -word {} -depth {} -fw.q {}".format(v, str((int(depth) * C)), default_q).split()).split()
    try:
        result_ids = [str(hash_to_id_dict[test[10 + (3*i)]]) for i in range((int(depth) * C))]
    except:
        print(test)
        missed += 1

    average_latency += float(test[-1][:test[-1].find("m")])
    #Find ann neighbors
    neighbors_ann = ' '.join(map(str, ann_index_intent.get_nns_by_item(int(k), int(depth)))).split()
    
    #Find kdtree neighbors
    vector = np.array(ann_index_intent.get_item_vector(int(k))).reshape(1, -1)
    _, indices = _kdTree.query(vector, k=int(depth))
    neighbors = ' '.join(map(str, indices)).split()

    average += len(set(neighbors).intersection(result_ids))
    print(average, len(set(neighbors).intersection(result_ids)))

    average_recall_ann += len(set(neighbors_ann).intersection(result_ids))

    if not index % 10:
        print(index)
        print("Average recall: {:.2f}".format(float(average) / (index + 1 - missed)))
        print("latency: {:.2f}" .format(average_latency / (index + 1 - missed)))
        print("Average recall ANN: {:.2f}".format(float(average_recall_ann) / (index + 1 - missed)))
                
print("Average Recall: {:.2f}, Average Latency: {}".format(float(average) / 10000, average_latency / 10000))
#os.system("target/appassembler/bin/ApproximateNearestNeighborSearch -input ../id_to_embeddings.txt -path intent-idx -encoding fw -word test_word")
#os.system("target/appassembler/bin/ApproximateNearestNeighborSearch -stored -path intent-idx -encoding fw -word test_word")
print("done")

