import networkx as nx 
"""returns a node's descendants up to a certain depth"""
def get_descendants(graph,node,threshold):
    paths=nx.single_source_shortest_path(graph,node,threshold)
    all_nodes=[]
    for key in paths:
        for step in paths[key]:
            if step not in all_nodes+[node]:
                all_nodes.append(step)
    return all_nodes