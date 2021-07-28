import pickle

f = pickle.load(open("pentip_graph.pkl", "rb"))

print(f.nodes)
print(f.edges)
print(f.degree)
