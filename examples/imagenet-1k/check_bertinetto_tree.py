import pickle, networkx as nx

# Load Bertinetto's tree
with open("wordnet_graph/Bertinetto/tiered_imagenet_tree.pkl", "rb") as f:
    bert_tree = pickle.load(f)  # this is an nltk.Tree

# Load our ctree
with open("wordnet_graph/ctree.gpickle", "rb") as f:
    our_ctree = pickle.load(f)

# Compare node counts
print("Bertinetto nodes:", len(bert_tree.leaves()), "leaves")  # or bert_tree.treepositions()
print("Our ctree nodes:", our_ctree.number_of_nodes())
print("Our ctree leaves:", sum(1 for n in our_ctree.nodes() if our_ctree.in_degree(n)==0))