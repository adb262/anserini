from annoy import AnnoyIndex
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embedding", nargs="?", default="intent")

PATH = ""
PATH_TO_EMBEDDINGS = "id_to_embeddings.txt"
PATH_ID_TO_HASH = "id_to_string_hash.json"
PATH_HASH_TO_ID = "string_hash_to_id.json"
PATH_TO_INTENT = "full_enc.intent.ann"
PATH_TO_GENERIC = "full_enc.generic.ann"

args = parser.parse_args()

def loadAnnIndex(filename : str, vector_size : int, metric : str) -> AnnoyIndex:
    """Load an Annoy index from file."""
    print("ANN_ loading ", filename)
    t = AnnoyIndex(vector_size, metric)
    t.load(filename)
    return t

def getAnnIndex(embedding_type) -> AnnoyIndex:
    params = {}
    if embedding_type == "intent":
        params["intent_embeddings"] = PATH + PATH_TO_INTENT
        n = 200
    else:
        params["intent_embeddings"] = PATH + PATH_TO_GENERIC
        n = 150

    return loadAnnIndex(params["intent_embeddings"], vector_size=n, metric="angular")

def getStringHash(_id : int) -> str:
    encoding = [chr(i) for i in np.random.randint(97, 123, size=50)]
    return ''.join(encoding)

def loadDataAndDumpToJson(ann_index : AnnoyIndex) -> None:
    data = [ann_index.get_item_vector(i) for i in range(ann_index.get_n_items()) if not np.isnan(np.sum(ann_index.get_item_vector(i)))]

    id_dict = {}
    hash_dict = {}

    with open(PATH_TO_EMBEDDINGS, 'w') as f:
        for _id, embedding in zip(list(range(ann_index.get_n_items())), data):
            string_hash = getStringHash(_id)
            f.write("{} {}\n".format(string_hash, ' '.join(str(val) for val in embedding)))
            id_dict[_id] = string_hash
            hash_dict[string_hash] = _id

    with open(PATH + PATH_ID_TO_HASH, "w") as f:
        json.dump(id_dict, f)

    with open(PATH + PATH_HASH_TO_ID, "w") as f:
        json.dump(hash_dict, f)

    print("dumped")

if __name__ == "__main__":
    embedding_type = args.embedding
    ann_index = getAnnIndex(embedding_type)

    loadDataAndDumpToJson(ann_index)


