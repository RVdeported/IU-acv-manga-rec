from annoy import AnnoyIndex
import numpy as np

#=================================================#
# AnnoyTree class                                 #
#=================================================#
# Realisation of AnnoyIndex KNN class for storing and 
# search of closest vector in the database
class AnnoyTree():
    def __init__(self, 
                 features: list[np.array], # list of 1D vectors with features
                 name:     list[str],      # list of corresponding Manga titles
                 img_path: list[str],      # list of manga location  
                 dist:     str='angular', 
                 trees:    int=10
                ):

        self.name = name
        self.img_path = img_path
        
        # creation of the Index trees and addition of items
        result_tree = AnnoyIndex(features[0].shape[0], dist)
        for idx in range(len(self.name)):
            result_tree.add_item(idx, features[idx].flatten())
        result_tree.build(trees)
        
        self.tree = result_tree
        
    #-------------------------------------------------#
    # Inference                                       #
    #-------------------------------------------------#
    def infer(self, img_feature, top_n=5, nns=10):
        best_idx, dists = self.tree.get_nns_by_vector(
            img_feature.flatten(), 
            nns, 
            include_distances=True)
        return list(set(map(lambda x: (self.name[x[0]], self.img_path[x[0]], x[1]), zip(best_idx, dists))))[:top_n]

