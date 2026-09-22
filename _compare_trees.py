import pickle

with open("examples/imagenet-1k/wordnet_graph/tree.gpickle", "rb") as f:
    old_tree = pickle.load(f)

with open("examples/imagenet/data/tree/tree.gpickle", "rb") as f:
    new_tree = pickle.load(f)

# old tree has child->parent edges, so nodes are the same set
old_nodes = set(old_tree.nodes())
new_nodes = set(new_tree.nodes())

print("In old but not new:", old_nodes - new_nodes)
print("In new but not old:", new_nodes - old_nodes)