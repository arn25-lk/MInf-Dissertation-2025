from abc import ABC, abstractmethod
import itertools
import random
import numpy as np
from typing import Iterable, List 
import networkx as nx
import matplotlib.pyplot as plt
import math

# from advanced_algorithms.decorators import collect_timestamps_on_call




class GeneralFairAllocationInstance:
    def __init__(self, N : int, M : int,  valuation_table : Iterable[str] = None, dist="uniform", params=[]):
        self.N = N
        self.M = M
        self.agents = set([i for i in range(1,self.N+1)])
        self.goods = set([i for i in range(1, self.M+1)])
        v_table = random_valuation_table(self.agents, self.goods, dist=dist, params=params) if valuation_table is None else valuation_table
        self.valuation_func = np.array(v_table)
        self.b_vals = [0 for _ in range(self.N)]
        self.allocation = None
        self.pool = None
        

    def get_val(self, n : int, m : int) -> float:
        return self.valuation_func[n-1][m-1]

    def add_good_from_pool(self, agent, good):
        self.allocation[agent].append(good)
        self.pool = self.pool - {good}
        self.b_vals[agent-1] += self.get_val(agent, good)
    
    def remove_good_from_allocation(self, agent, good):
        self.allocation[agent].remove(good)
        self.pool = self.pool | {good}
        self.b_vals[agent-1] -=  self.get_val(agent, good)

    def assign_bundle_in_allocation(self, agent, bundle : list[int]):
        """
        Assigns a bundle of goods to an agent.

        Parameters:
        agent (int): The index of the agent.
        bundle (list): A list of goods to assign to the agent.

        Returns:
        None
        """
        self.allocation[agent] = bundle 
        self.b_vals[agent-1] = sum([self.get_val(agent, good) for good in bundle])
    

    def gen_allocations(self, type='empty', allocation_dict=None) -> dict:
        """
        Generates allocations based on the type of allocation.

        Parameters:
        data (list of str, optional): List of strings of the form 'good->agent'. Defaults to None.
        type (str, optional): Type of allocation. Defaults to 'empty'.
        allocation_dict (dict, optional): Dictionary specifying the allocation. Required if type is 'from_dict'. Defaults to None.

        Returns:
        dict: A dictionary representing the allocation of goods to agents.

        Raises:
        AttributeError: If type is 'from_dict' and allocation_dict is None.
        """


        if type == 'empty':
            self.allocation, self.pool = {a : [] for a in range(1, self.N +1)}, set([i for i in self.goods])
            
        if type == 'random':
            allocation = {a: [] for a in range(1, self.N + 1)}
            for g in self.goods:
                agent = random.randint(1, self.N)  # Assign each good to a random agent
                allocation[agent].append(g)

            self.allocation, self.pool = allocation, {}
        if type == 'random_partial':
            allocation = {a: [] for a in range(1, self.N + 1)}
            goods = list(self.goods)
            number = random.randint(1, self.M)
            alloc_goods = set(random.sample(goods, number))
            pool = self.goods - alloc_goods
            for g in alloc_goods:
                agent = random.randint(1, self.N)
                allocation[agent].append(g)

            self.allocation, self.pool = allocation, pool
        if type == 'from_dict':
            if allocation_dict is None:
                raise AttributeError
            alloc_goods = []
            for a, gs in allocation_dict.items():
                alloc_goods.extend(gs)
                self.b_vals[a-1] = sum([self.get_val(a, g) for g in gs])
            pool = self.goods - set(alloc_goods)
            self.allocation, self.pool = allocation_dict, pool
        
    def bundle_value(self, i, j) -> float:
        '''
        Value of i's bundle w.r.t j's valuation function
        '''
        if i == j:
            return self.b_vals[i-1]
        bundle_value = 0 
        for g in self.allocation[i]:
            bundle_value += self.get_val(j,g)
        return bundle_value
    
    def get_envy_graph(self, debug=False):
        if self.allocation == None: 
            return
        G = nx.DiGraph()
        G.add_nodes_from(self.agents)
        for i in self.agents:
            bundle_i = self.bundle_value(i,i)
            for j in self.agents - {i}:
                bundle_j = self.bundle_value(j, i)
                # relatively close  inequality test 
                if bundle_i < bundle_j:
                    G.add_edge(i,j)
                    if debug:
                        G.add_edge(i,j,X_i=self.allocation[i], X_j=self.allocation[j])
        return G
    
    def get_reduced_graph(self, alpha=(2/3), debug=False):
        if self.allocation == None: 
            return
        G = nx.DiGraph()
        G.add_nodes_from(self.agents)
        for i in self.agents:
            bundle_i = self.bundle_value(i,i)
            for j in self.agents - {i}:
                bundle_j = self.bundle_value(j,i)    
                if len(self.allocation[i]) == 2 and len(self.allocation[j]) == 1:
                    if bundle_i < alpha*bundle_j:
                        G.add_edge(i,j)
                elif bundle_i < bundle_j:
                    G.add_edge(i,j)
                    if debug:
                        G.add_edge(i,j,X_i=self.allocation[i], X_j=self.allocation[j])
        return G

    def get_enhanced_graph(self, alpha=(2/3), debug=False):
        G = self.get_reduced_graph(debug=debug)
        for s in self.get_sources(G):
            if len(self.allocation[s]) != 2:
                continue
            for i in self.agents:
                if len(self.allocation[i]) == 1: 
                    bundle_i = self.bundle_value(i,i)
                    bundle_s = self.bundle_value(s,i)
                    if bundle_s >  alpha*bundle_i or bool(np.isclose(bundle_i, alpha*bundle_s, atol=1e-7)):
                        G.add_edge(i,s) 
                        if debug :
                            G.add_edge(i,s,X_i=self.allocation[i], X_s=self.allocation[s])
        return G
     
    def get_max_val_good(self, pool, agent):
        if not pool:
            return None
        x = -1
        g_star = None
        for g in pool:
            val = self.get_val(agent, g)
            if val > x:
                g_star, x = g, val
        return g_star

    
    def round_robin(self, order: list = None, max_iter: int = None):
        """
        Executes the round robin algorithm for fair allocation of goods among agents.

        Parameters:
        order (list, optional): A list specifying the order in which agents will be allocated goods. 
                                If None, the default order of agents will be used.
        max_iter (int, optional): The maximum number of iterations to run the algorithm. 
                                  If None, the algorithm will run until all goods are allocated.

        Returns:
        None
        """
        i = 0
        iterations = 0
        left_order = order
        if left_order is None:
            left_order = sorted(list(self.agents))
        while len(self.pool) > 0:
            g = self.get_max_val_good(self.pool, left_order[i])
            self.add_good_from_pool(left_order[i], g)
            # self.allocation[left_order[i]].append(g)
            # self.pool -= {g}
            
            i = (i + 1) % len(left_order) 
            if i == 0:
                iterations +=1
            if iterations == max_iter:
                break

    def contains_cycles(self, G : nx.DiGraph):
        try:
            return len(list(nx.find_cycle(G, orientation='original'))) > 0
        except nx.NetworkXNoCycle:
            return False


    def resolve_all_cycles(self, graph : nx.DiGraph, type='ev'):
        G = graph
        while self.contains_cycles(G):
            # cycle resolution
            edges = list(nx.find_cycle(G, orientation='original'))
            first_alloc = self.allocation[edges[0][0]]

            for i, j, _ in edges:
                self.assign_bundle_in_allocation(i, self.allocation[j])

            agent = edges[-1][0]
            self.assign_bundle_in_allocation(agent, first_alloc)
            match type:
                case 'ev':
                    G = self.get_envy_graph()
                case 'r':
                    G = self.get_reduced_graph()
                case 'en':
                    G = self.get_enhanced_graph()
        return G
    # runtime
    # frequency of steps 
    # seed differences
    def get_sources(self, G) -> List[int] | None:
        unenvied_agents = [i for i in self.agents if G.in_degree[i] == 0]
        return unenvied_agents if len(unenvied_agents) > 0 else None
    

    def get_valuable_goods(self) -> dict[int, List[int]]:
        return self.get_critical_goods(eps=3)
    def get_critical_goods(self, eps=2) -> dict[int, List[int]]: 
        has_critical = {}
        for g in self.pool:
            for i in self.agents:
                vi_Ai = self.bundle_value(i,i)
                vi_g = self.get_val(i,g)
                if vi_Ai < eps*vi_g:
                    if g in has_critical:
                        has_critical[g].append(i)
                    else: 
                        has_critical[g] = [i]
        return has_critical
    
    def path_resolution(self,path):
        path = list(nx.utils.pairwise(path))
        f_alloc = self.allocation[path[0][0]]
        for i, j in path:
            self.assign_bundle_in_allocation(i, self.allocation[j])
        self.assign_bundle_in_allocation(path[-1][1], [])
        return f_alloc
    
    
    def path_resolution_star(self, path, i):
        src_bundle = self.path_resolution(path)
        g_src = self.get_max_val_good(src_bundle, i) 
        g_star = self.get_max_val_good(self.pool, i)
        
        new_bundle = [g_src, g_star]
        self.assign_bundle_in_allocation(i, new_bundle)
        self.pool = self.pool - set(new_bundle)

    def champions(self, i, j, g, epsilon=1/3):
        """
        Check if agent i is a champion for agent j over good g.

        Parameters:
        i (int): The index of the first agent.
        j (int): The index of the second agent.
        g (int): The index of the good.

        Returns:
        bool: True if agent i is a champion for agent j over good g, False otherwise.
        """
        b = self.allocation[j].copy()
        b.append(g)
        bundle_set = set(b)
        bundle_i = self.bundle_value(i,i)

    
        for S in [set(s) for k in range(2, len(bundle_set)+1) for s in itertools.combinations(bundle_set, k)]:
            val_i_S = sum(self.get_val(i, g) for g in S)
            if bundle_i >= (1-epsilon)*val_i_S:
                continue
            condition_2 = True
            for a in self.agents:
                s_Val = sum(self.get_val(a,g) for g in S)
                for k in S:
                    val_a_Xa = self.bundle_value(a,a)
                    val_a_S = s_Val - self.get_val(a,k)
                    if  (1-epsilon)*val_a_S > val_a_Xa:
                        condition_2 = False
                        break
                if not condition_2:
                    break
            if condition_2:
                return True
        return False
    
    def finds_valuable(self, agent, good, epsilon=1/3):
      return self.get_val(agent, good) >  (epsilon * self.bundle_value(agent, agent))
    
        
    def resolve_cycle(self, edges, type='ev'):
        f_alloc = self.allocation[edges[0][0]]
        for i, j, _ in edges:
            self.assign_bundle_in_allocation(i, self.allocation[j])

        self.assign_bundle_in_allocation(edges[-1][0], f_alloc)
        match type:
            case 'ev':
                return self.get_envy_graph()
            case 'r':
                return self.get_reduced_graph()
            case 'en':
                return self.get_enhanced_graph()
    
  #  @collect_timestamps_on_call
    def envy_cycle_elimination(self, G, type='ev', return_graphs=False):
        """
        ECE algorithm on arbitrary graph instances.

        Parameters:
        G (nx.DiGraph): The input graph.
        type (str, optional): Type of graph to use. Can be 'ev' for envy graph, 'r' for reduced graph, 'en' for enhanced graph. Defaults to 'ev'.
        return_graphs (bool, optional): Specifies if the graphs should be returned. Defaults to False.

        Returns:
        list: A list of graphs if return_graphs is True, otherwise None.
        """
        graphs = []
        while len(self.pool) > 0:
            if return_graphs:
                graphs.append(G)
                # nx.draw(G,pos=nx.spring_layout(G))
            G = self.resolve_all_cycles(G, type=type)
            # when cycles have been resolved there must raise a source
            unenvied_agents = [i for i in self.agents if G.in_degree[i] == 0]
            i = unenvied_agents[0]
            
            g = self.get_max_val_good(self.pool, i)
            self.add_good_from_pool(i, g)

            G = self.get_envy_graph()
        if return_graphs:
            return graphs
    

def random_valuation_table(N, M, r=(0,1000),  dist='uniform', params=[]):
    """
    dist : str
        The distribution to use. Options are 'uniform', 'binary', 'quadratic_face_up', 'quadratic_face_down', 'uniform_hom'.
    r : tuple
        The range of values to use.
    params : list
        A list of parameters to use for the distribution.   
    """ 
    match dist:
        case "uniform":
            lo, hi  = r
            valuation_table = np.random.uniform(lo, hi, size=(len(N), len(M)))
            return valuation_table
        case "binary":
            valuation_table = np.random.choice([0,1], size=(len(N), len(M)))
            return valuation_table
        case "quadratic_face_down":
            a, b = r
            alpha = 12/((b-a)**3)
            beta = (a + b)/2
            support = np.arange(a, b + 1)
            raw_pdf = np.array([-alpha * (x - beta)**2 for x in support])
            raw_pdf -= raw_pdf.min()  # Shift so minimum is zero
            raw_pdf /= raw_pdf.sum()  # Normalize to probabilities
            valuation_table = [[np.random.choice(support, p=raw_pdf) for _ in M] for _ in N]
            return valuation_table
        case "quadratic_face_up":
            a, b = r
            alpha = 12/((b-a)**3)
            beta = (a + b)/2
            f = lambda x : -1*alpha*(x - beta)**2
            dists = [f(i) for i in range(a, b+1)]
            dists /= np.sum(dists)
            valuation_table = [[np.random.choice(range(a, b+1), p=dists) for _ in M] for _ in N]
            return valuation_table
        case "uniform_homogenous":
            # make every agent have the same valuation for every good by drawing M goods from uniform distribution and replicating N times
            lo, hi  = r
            valuation_table = np.random.uniform(lo, hi, size=(len(M),))
            valuation_table = np.int32(valuation_table)
            valuation_table = np.tile(valuation_table, (len(N), 1))
            # make sure all values are positive
            valuation_table = np.abs(valuation_table)
            return valuation_table
        
        case "normal":
            if len(params) != 2:
                raise ValueError("params must contain mu and sigma")
            if params[0] < 0 or params[1] < 0:
                raise ValueError("mu and sigma must be positive")

            mu, sigma = params[0], params[1]
            valuation_table = np.random.normal(mu, sigma, size=(len(N), len(M)))
            # make sure all values are positive
            valuation_table = np.abs(valuation_table)
            # make sure all values are floating point with 3 decimal precision
            valuation_table = np.round(valuation_table, 3)

            return valuation_table


class MultiGraphInstance (GeneralFairAllocationInstance):
    def __init__(self, N : int, M : int, valuation_table=None, dist="uniform", params=[]): 
        if valuation_table is not None:
            raise ValueError("Custom Valuation table not supported for MultiGraph.")
        self.N = N
        self.M = M
        self.agents = set([i for i in range(1,self.N+1)])
        self.goods = set([i for i in range(1, self.M+1)])
        v_table = random_valuation_table(self.agents, self.goods, dist=dist, params=params) if valuation_table is None else valuation_table
        self.valuation_func = self.get_random_multigraph_valuations(v_table)
        self.b_vals = [0 for _ in range(self.N)]
        self.allocation = None
        self.pool = None
    
    def erdos_renyi_GnN(self):
        """
        Generates a random Erdos-Renyi graph with n (number of agents) nodes and m (number of goods) edges.
        """
        G = nx.MultiGraph()
        G.add_nodes_from(self.agents)
        # Generate random edges between agents and goods
        # using combinations with replacement to allow multiple edges
        samples = random.choices(
            list(itertools.combinations_with_replacement(self.agents, 2)),
            k=len(self.goods)
        )
        G.add_edges_from([(i,j,{"g": g}) for (i,j), g in zip(samples, self.goods)])
        return G

    # sample a pair and assign to an edge 
    def get_random_multigraph_valuations(self, v_table, debug=False):
        G = self.erdos_renyi_GnN()
        if debug:
            plot_graphs([G], pos=None)
        mat = np.zeros((self.N, self.M))
        for u,v, key, data in G.edges(data=True, keys=True):
            g = data['g']
            mat[u-1][g-1] = v_table[u-1][g-1]
            mat[v-1][g-1] = v_table[v-1][g-1]
        return mat

class TopNInstance(GeneralFairAllocationInstance):
    def __init__(self, N : int, M : int, valuation_table=None, dist="uniform", params=[]): 
        if valuation_table is not None:
            raise ValueError("Custom Valuation table not supported for Top-N.")
        self.N = N
        self.M = M
        self.agents = set([i for i in range(1,self.N+1)])
        self.goods = set([i for i in range(1, self.M+1)])
        self.top_n = set(random.sample(sorted(self.goods), self.N))
        self.b_vals = [0 for _ in range(self.N)]
        if dist == 'binary':
            self.valuation_func = [[1 if g+1 in self.top_n else 0 for g in range(self.M)] for _ in range(self.N)]
        else:
            top_n_valuations = random_valuation_table(self.agents, self.goods, r=(1,1000), dist=dist, params=params)
            b_valuations = []
            for idx, oc in enumerate(top_n_valuations):
                x = min(oc)
                b_valuations.append( random_valuation_table([idx], self.goods, r=(0,x), dist=dist, params=params)[0])
            self.concat_valuation_tables(self.top_n, top_n_valuations, b_valuations)
        self.allocation = None
        self.pool = None

    def get_top_n(self):
        return self.top_n
    
    def concat_valuation_tables(self, top_n, top_n_valuations, b_valuations):
        valuation_table = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in range(self.M):
                if j+1 in top_n:
                    valuation_table[i][j] = top_n_valuations[i][j]
                else:
                    valuation_table[i][j] = b_valuations[i][j]
        self.valuation_func = valuation_table


def plot_graphs(graphs, pos=None, labels=None, save_path=None):
        n_graphs = len(graphs)
        cols = 3  # Number of columns in the grid
        rows = math.ceil(n_graphs / cols)  # Calculate rows dynamically
        plt.show()
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        # Flatten axes for easy iteration (even if there's only one row)
        axes = axes.flatten()     # Plot each graph
        for i, graph in enumerate(graphs):
            if pos is None:
                pos_x = nx.spring_layout(graph)
            else:
                pos_x = pos
            nx.draw(graph, node_color='lightblue', ax=axes[i], pos=pos_x, with_labels=True)
            edge_labels = {
            (u, v): data
            for u, v, data in graph.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
            graph,
            pos=pos_x,
            edge_labels=edge_labels,
            ax=axes[i],
            font_size=8
            )
            if labels:
                axes[i].set_title(labels[i])
            else:
                axes[i].set_title(f"i = {i+1}")
        # Hide unused subplots (if any)
        for j in range(len(graphs), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        # Adjust layout
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()



if __name__ == '__main__':
    # Example usage
    G = GeneralFairAllocationInstance(N=8, M=30, dist="uniform_homogenous")
    
    