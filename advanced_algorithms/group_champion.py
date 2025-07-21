import itertools
import networkx as nx
from collections import deque

from advanced_algorithms import indivGoodInstance as igi
from advanced_algorithms import advAlgorithms as alg




 
# D is the maximum number of agents that can find a good valuable
# in the group champion graph
D = 4

def source_map(envy_graph):
    """
    Given an envy graph represented as a NetworkX DiGraph, this function finds the group champion graph.
    
    Args:
        envy_graph (nx.DiGraph): A directed graph where nodes represent entities and edges represent envy relationships.
    
    Returns:
        dict: A hashmap where each node is assigned a source such that there is a path from the source to the node.
    """
    # Step 1: Find all sources in the envy graph (nodes with no incoming edges)
    sources = [node for node in envy_graph.nodes if envy_graph.in_degree(node) == 0]
    
    # Step 2: Perform BFS from each source to find reachable nodes
    source_to_node_map = {}
    for source in sources:
        visited = set()
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                source_to_node_map[current] = source
                for neighbor in envy_graph.successors(current):
                    queue.append(neighbor)
    # Step 3: Return the source-to-node mapping
    return source_to_node_map



def get_vertex_set(gen: igi.GeneralFairAllocationInstance, source_map):
    
    valuable_goods = {g : [] for g in gen.pool}
    for g in gen.pool:
        for a in gen.agents:
            if gen.finds_valuable(a, g):
                valuable_goods[g].append((a,source_map[a]))
    # remove the goods that are valuable to more than D agents
    # and return the remaining goods
    to_remove = [g for g in valuable_goods if len(valuable_goods[g]) > D]
    for g in to_remove:
        del valuable_goods[g]
    return valuable_goods



def get_group_champion_graph(gen : igi.GeneralFairAllocationInstance):
    ev_graph = gen.get_envy_graph()
    s_map = source_map(ev_graph)
    V = get_vertex_set(gen, s_map)

    G_champ = nx.DiGraph()
    V_set = [v for v in V]
    for g in V_set:    
        for a,s_a in V[g]:
            G_champ.add_node((g, s_a), component=s_a)

    for g, h in itertools.product(V_set, V_set):
        # If the goods are not the same, check for envy relationships
        # between the agents and the goods
        # and add edges accordingly
        if g != h:
            for i, j in itertools.product(V[g], V[h]):
                a, s_a = i
                b, s_b = j
                if gen.champions(a, s_b, g):
                    G_champ.add_edge((g, s_a), (h, s_b))
    return G_champ, len(V_set)
    
    
    

def layout_by_component(G):
    # Group nodes by component
    from collections import defaultdict

    component_nodes = defaultdict(list)
    for node in G.nodes:
        g, s_a = node
        component_nodes[g].append(node)

    # Create layout: one column per component
    pos = {}
    x_spacing = 3  # horizontal spacing between components
    y_spacing = 1  # vertical spacing within a component

    for i, (component, nodes) in enumerate(sorted(component_nodes.items())):
        for j, node in enumerate(nodes):
            pos[node] = (i * x_spacing, -j * y_spacing)

    return pos

# Example usage
if __name__ == "__main__":
    # Example envy graph using NetworkX DiGraph
    for i in range(100):
        G = igi.GeneralFairAllocationInstance(N=8, M=30, dist="uniform_homogenous")
        T = alg.ThreePA(G)
        T.run()
        # Assuming G is an instance of GeneralFairAllocationInstance
        envy_graph = G.get_envy_graph()
        
        group_champion_graph = get_group_champion_graph(G)
        pos = layout_by_component(group_champion_graph)

        # Draw the envy graph
        igi.plot_graphs([group_champion_graph], pos=pos)
        print(G.allocation)
