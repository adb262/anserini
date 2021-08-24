import json
import numpy as np
from annoy import AnnoyIndex
import os
import subprocess
import sys
import random
from sklearn.neighbors import KDTree
import argparse
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings", nargs="?", default="intent")
parser.add_argument("--c", nargs="?", default="50", type=int,
    help="An integer to define our search space BEFORE resorting"
    )
parser.add_argument("--q", nargs="?", default="100", type=int, help="Quantization factor")
parser.add_argument("--k", nargs="?", default="100", type=int, help="K nearest neighbors desired")
parser.add_argument("--build", nargs="?", default="False", type=str2bool,
    help="Pass in True to build an index, False if it already exists"
    )

args = parser.parse_args()
PATH = "../"
PATH_TO_EMBEDDINGS = "id_to_embeddings.txt"
PATH_ID_TO_HASH = "id_to_string_hash.json"
PATH_HASH_TO_ID = "string_hash_to_id.json"
PATH_TO_INTENT = "full_enc.intent.ann"
PATH_TO_GENERIC = "full_enc.generic.ann"
INDEX_PATH = "intent-idx-stored"

def loadAnnIndex(filename : str, vector_size : int, metric : str) -> AnnoyIndex:
    """Load an Annoy index from file."""
    print("ANN_ loading ", filename)
    t = AnnoyIndex(vector_size, metric)
    t.load(filename)
    return t

def getAnnIndex(embedding_type) -> AnnoyIndex:
    """
    Load the appropriate AnnoyIndex based on the embedding type

    Return AnnoyIndex
    """
    params = {}
    if embedding_type == "intent":
        params["intent_embeddings"] = PATH + PATH_TO_INTENT
        n = 200
    else:
        params["intent_embeddings"] = PATH + PATH_TO_GENERIC
        n = 150

    return loadAnnIndex(params["intent_embeddings"], vector_size=n, metric="angular")

def loadDataAndCreateTree(ann_index : AnnoyIndex) -> KDTree:
    """
    Load the data from the AnnoyIndex and put into a KDTree

    Return the KDTree
    """
    data = [ann_index.get_item_vector(i) for i in range(ann_index.get_n_items()) if not np.isnan(np.sum(ann_index.get_item_vector(i)))]
    _kdTree = KDTree(np.array(data))
    return _kdTree

def loadHashDictionaries():
    """
    Load dictionaries containing id -> hash and hash -> id mappings
    
    These dictionaries are essential due to some restrictive properties
    of the anserini repository

    Return both dictionaries
    """
    with open(PATH + PATH_ID_TO_HASH, "r") as f:
        id_to_hash_dict = json.load(f)

    with open(PATH + PATH_HASH_TO_ID, "r") as f:
        hash_to_id_dict = json.load(f)
    
    return id_to_hash_dict, hash_to_id_dict

def buildIndex(build, q) -> None:
    """
    Build a local Lucene inverted index with the specified quantization factor
    """
    if build:
        os.system("rm -r intent-idx-stored")
        print("Removed index, preparing to build")
        os.system(
            "target/appassembler/bin/IndexVectors -input {} -path {} -encoding fw -stored -fw.q {}".format(PATH + PATH_TO_EMBEDDINGS, INDEX_PATH, q)
            )

def runTests(
    depth : int, 
    c : int, 
    q : int, 
    id_to_hash_dict, 
    hash_to_id_dict, 
    ann_index : AnnoyIndex, 
    _kdTree : KDTree
    ) -> None:
    """
    Run tests to collect the average recall and latency for ANN and 
    Fake Words approaches.  Compare to the KDTree.
    """
    shuffled_id_dict = list(id_to_hash_dict.items())
    random.shuffle(shuffled_id_dict)

    average, average_latency = 0, 0
    average_recall_ann, average_latency_ann = 0, 0
    missed = 0
    for index, (k, v) in enumerate(shuffled_id_dict):
        if index >= 10000:
            break

        test = subprocess.check_output(
            "target/appassembler/bin/ApproximateNearestNeighborSearch -stored -path {} -encoding fw -word {} -depth {} -fw.q {}".format(
                INDEX_PATH, 
                v, 
                str((depth * c)), 
                q).split()
            ).split()
        try:
            result_ids = [str(hash_to_id_dict[test[10 + (3*i)].decode("utf-8")]) for i in range((depth * c))]
        except:
            print("Poor results: ", test)
            missed += 1
            continue
        average_latency += float(test[-1].decode("utf-8")[:test[-1].decode("utf-8").find("m")])

        #Find ann neighbors
        neighbors_ann = ' '.join(map(str, ann_index.get_nns_by_item(int(k), depth))).split()

        #Find kdtree neighbors
        start = time.time()
        vector = np.array(ann_index.get_item_vector(int(k))).reshape(1, -1)
        _, indices = _kdTree.query(vector, k=depth)
        neighbors = ' '.join(map(str, indices))[1:-1].split()
        average_latency_ann += time.time() - start

        average += len(set(neighbors).intersection(result_ids))
        print(average, len(set(neighbors).intersection(result_ids)))

        average_recall_ann += len(set(neighbors_ann).intersection(result_ids))

        if not index % 10:
            print(index)
            print("Average recall: {:.2f}".format(float(average) / (index + 1 - missed)))
            print("latency: {:.2f}" .format(average_latency / (index + 1 - missed)))
            print("Average recall ANN: {:.2f}".format(float(average_recall_ann) / (index + 1 - missed)))
            print("Average latency ANN: {:.2f}".format(float(average_latency_ann) / (index + 1 - missed)))

    print("Average Recall: {:.2f}, Average Latency: {}".format(float(average) / 10000, average_latency / 10000))
    print("done")


if __name__ == "__main__":
    embedding_type, build, q, c, depth = args.embeddings, args.build, args.q, args.c, args.k
    print(embedding_type, build, q, c, depth)
    ann_index = getAnnIndex(embedding_type)
    _kdTree = loadDataAndCreateTree(ann_index)

    id_to_hash_dict, hash_to_id_dict = loadHashDictionaries()
    buildIndex(build, q)

    runTests(depth, c, q, id_to_hash_dict, hash_to_id_dict, ann_index, _kdTree)


                        
