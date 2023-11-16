from annoy import AnnoyIndex
import numpy as np
import pickle

class AnnoyTree():
    def __init__(self, features, name, img_path, dist='angular', trees=10):
        self.name = name
        self.img_path = img_path
        
        
        # print(images)
        result_tree = AnnoyIndex(features[0].shape[0], dist)
        for idx in range(len(self.name)):
            result_tree.add_item(idx, features[idx].flatten())
        
        result_tree.build(trees)
        
        self.tree = result_tree
        
    
    def infer(self, img_feature, top_n=5, nns=10):
        best_idx, dists = self.tree.get_nns_by_vector(img_feature.flatten(), nns, include_distances=True)
        return list(set(map(lambda x: (self.name[x[0]], self.img_path[x[0]], x[1]), zip(best_idx, dists))))[:top_n]
    
    # def save(self, annoypath='annnoy.ann', clpath='annoy.pick'):
    #     self.tree.save(annoypath)
    #     tree = self.tree
    #     self.tree = None
    #     with open(clpath, "wb+") as f:
    #         pickle.dump(self, f)
    #     self.tree = tree
            
    # @staticmethod
    # def load(dimensionality, annoypath='annoy.ann', clpath='annoy.pik', dist='angular'):
    #     with open(clpath, "rb") as f:
    #         annoy = pickle.load(f)
    #     idx = AnnoyIndex(dimensionality, dist)
    #     idx.load(annoypath)
    #     annoy.tree = idx
    #     return annoy
