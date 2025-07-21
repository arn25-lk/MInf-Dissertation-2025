import os, sys

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from advanced_algorithms.indivGoodInstance import GeneralFairAllocationInstance, TopNInstance, MultiGraphInstance, plot_graphs
import math
import networkx as nx
from functools import cmp_to_key

from advanced_algorithms.decorators import collect_alg_timestamps_on_call, collect_frequency_on_call, collect_timestamps_on_call

from abc import ABC

class Algorithm(ABC):
    def run(self):
        pass

class DraftAndEliminate(Algorithm):
    def __init__(self, gen : GeneralFairAllocationInstance, debug=False):
        self.N = gen.N
        self.M = gen.M
        self.G = gen
        self.G.gen_allocations(type='empty')
    
    def _arg_max(self, H ,R, i):
        j = i
        max_val = -math.inf
        for t in R:
            h_t = H[t]
            val = self.G.get_val(i, h_t)
            if val > max_val:
                j = t
                max_val = val
        return j
    
    @collect_timestamps_on_call
    def _preprocessing(self) -> tuple:
        A = self.G.agents.copy()
        M = self.G.pool.copy()
        A_order = sorted(A)
        t = [0]*self.N
        L = set()
        l_order = []
        if len(M) < len(A):
            return A_order, len(L)
        H = {}
        phi = ( 1 + math.sqrt(5) ) / 2
        while len(A_order) != 0: 
            i = A_order[0]
            H[i] = self.G.get_max_val_good(M, i)
            timestamp = self.M - len(M) + 1
            t[i-1] = timestamp
            R = (self.G.agents - (A | L)) | {i}
            j = self._arg_max(H,R,i)
            if phi * self.G.get_val(i, H[i]) < self.G.get_val(i, H[j]):
                H[i] = H[j]
                L |= {i}
                l_order.append(i)
                A = (A - {i}) | {j}
                A_order = sorted(A)
            else:
                A -= {i}
                A_order.pop(0)
                M.discard(H[i])
        K = list(self.G.agents - L)
        K = sorted(K, key=cmp_to_key(lambda i, j: t[i-1] - t[j-1]))
        l_order.extend(K)
        return l_order, len(L)
    
    @collect_alg_timestamps_on_call
    def run(self):
        l_order, n_prime = self._preprocessing()
        self.G.round_robin(order=l_order, max_iter=1)
        rev_order =  l_order[::-1][:len(l_order) - n_prime]
        self.G.round_robin(order=rev_order, max_iter=1)
        envy_graph = self.G.get_envy_graph()
        self.G.envy_cycle_elimination(envy_graph, type='ev')

class ThreePA(Algorithm):
    
    def __init__(self, gen : GeneralFairAllocationInstance, seed=None, plus=False):
        self.N = gen.N
        self.M = gen.M
        self.G = gen
        self.plus = False
        if seed != None:
            self.G.gen_allocations(type='from_dict', allocation_dict=seed)
        else: 
            self.G.gen_allocations(type='empty')
            self.G.round_robin(max_iter=1)
        if plus:
            self.plus = True
        

    @collect_timestamps_on_call
    @collect_frequency_on_call
    def step1(gen : GeneralFairAllocationInstance):
        for i in gen.agents:
            if len(gen.allocation[i])==1:
                l = [g for g in gen.pool if gen.get_val(i, g) > gen.bundle_value(i,i)]
                if len(l) > 0:
                    return i, l[0]
        return None
    
    @collect_timestamps_on_call
    @collect_frequency_on_call
    def step2(gen : GeneralFairAllocationInstance):
        for i in gen.agents:
            if len(gen.allocation[i])==2:
                l = [g for g in gen.pool if gen.get_val(i, g) > (3/2)*gen.bundle_value(i,i)]
                if len(l) > 0:
                    return i, l[0]
        return None
    
    @collect_timestamps_on_call
    @collect_frequency_on_call
    def step3(gen : GeneralFairAllocationInstance):
        for i in gen.agents:
        
            if len(gen.allocation[i])==1:
                l = [(g_1,g_2) for g_1 in gen.pool for g_2 in gen.pool - {g_1} 
                     if gen.get_val(i, g_1) + gen.get_val(i,g_2)  > (2/3)*gen.bundle_value(i,i)]
                if len(l) > 0:
                    return i, l[0][0], l[0][1]
        return None
    
    @collect_timestamps_on_call
    @collect_frequency_on_call
    def step4(gen : GeneralFairAllocationInstance):
        for i in gen.agents:
            if len(gen.allocation[i])==2:
                l = [(g,g_prime) for g in gen.pool for g_prime in gen.allocation[i]
                     if gen.get_val(i, g) > gen.get_val(i,g_prime) ]
                if len(l) > 0:
                    return i, l[0][0], l[0][1]
        return None
    
    @collect_timestamps_on_call
    @collect_frequency_on_call
    def step5(gen : GeneralFairAllocationInstance, G_r):
        return gen.contains_cycles(G_r)

    @collect_timestamps_on_call
    @collect_frequency_on_call
    def step6(gen : GeneralFairAllocationInstance, G_r):
        if len(gen.pool) == 0: 
            return None
        sources = [i for i in gen.get_sources(G_r) if len(gen.allocation[i])==1]
        if len(sources) > 0:
            s = sources[0]
            g = gen.get_max_val_good(gen.pool, s)
            return s, g
        return None
    
    @collect_timestamps_on_call
    @collect_frequency_on_call
    def step7(gen : GeneralFairAllocationInstance, G_r):
        l = [(g,i) for g in gen.pool for i in gen.agents if (gen.get_val(i, g) > (2/3)*gen.bundle_value(i,i)) 
             and len(gen.allocation[i])==1]
        if len(l) == 1:
            g, i = l[0]
            sources = [s for s in gen.agents if G_r.in_degree[s] == 0 and nx.has_path(G_r,s,i)]
            if len(sources) > 1:
                return sources[0], i
        return None

            
    @collect_timestamps_on_call    
    @collect_frequency_on_call    
    def step8(gen : GeneralFairAllocationInstance, G_e):
        return gen.contains_cycles(G_e)
    
    @collect_timestamps_on_call    
    @collect_frequency_on_call    
    def step8_and_half(gen : GeneralFairAllocationInstance, G_e):
        sources = gen.get_sources(G_e)
        for s in sources:
            # all sources in G_e have in-degree 0 and size 2 bundles by the point this function is called
            for i in gen.agents - {s}:
                if not nx.has_path(G_e, s, i):
                    continue
                if gen.allocation[s]:
                    max_v_ig = max(gen.get_val(i, g_k) for g_k in gen.allocation[s])
                else:
                    max_v_ig = 0
                for g in gen.pool:
                    if gen.get_val(i, g) + max_v_ig > gen.bundle_value(i,i):
                        return s, i
        return None
    

    def run(self):
        gen = self.G
        while(True):
            res = ThreePA.step1(gen)
            if res != None:
                i, g = res
                gen.pool = (gen.pool | set(gen.allocation[i])) - {g}
                gen.assign_bundle_in_allocation(i, [g])
                # gen.allocation[i] = [g]
                continue
            res = ThreePA.step2(gen)
            if res != None:
                i,g = res
                gen.pool = (gen.pool | set(gen.allocation[i])) - {g}
                gen.assign_bundle_in_allocation(i, [g])
                # gen.allocation[i] = [g]
                continue
            res = ThreePA.step3(gen)
            if res != None:
                i, g_1, g_2 = res
                gen.pool = (gen.pool | set(gen.allocation[i])) - {g_1,g_2}
                gen.assign_bundle_in_allocation(i, [g_1,g_2])
                # gen.allocation[i] = [g_1,g_2]
                continue
            res = ThreePA.step4(gen)
            if res != None:
                i, g, g_prime = res
                before = gen.allocation[i].copy()
                gen.add_good_from_pool(i,g)
                gen.remove_good_from_allocation(i,g_prime)
                continue
            G_r = gen.get_reduced_graph()
            res = ThreePA.step5(gen, G_r)
            if res:
                gen.resolve_all_cycles(G_r, type='r')
                continue
            res = ThreePA.step6(gen, G_r)
            if res != None:
                s, g = res
                gen.pool = gen.pool - {g}
                gen.add_good_from_pool(s, g)
                continue
            res = ThreePA.step7(gen, G_r)
            if res != None:
                # Presumably, there is only one such path at this point 
                s, i = res
                paths = nx.all_simple_paths(G_r, s, i)
                for path in paths:
                    gen.path_resolution_star(path, i)
                    break
                continue
            G_e = gen.get_enhanced_graph()
            res = ThreePA.step8(gen, G_e)
            if res:
                gen.resolve_all_cycles(G_e, type='en')
                continue
            if self.plus:
                res = ThreePA.step8_and_half(gen, G_e)
                if res != None:
                    s, i = res
                    paths = nx.all_simple_paths(G_e, s, i)
                    for path in paths:
                        gen.path_resolution_star(path, i)
                        break
                    continue
            break
    @collect_timestamps_on_call
    def uncontested_critical(gen : GeneralFairAllocationInstance, type='en'):
        graph = gen.get_enhanced_graph()
        graph_e = gen.resolve_all_cycles(graph,type)

        l = gen.get_critical_goods()
        while len(l) > 1:
            # while there exists an agent i and a good g_i such that v_i(g_i) > 1/2 * v_i(i)
            i, g_i = l.pop()
            sources = gen.get_sources(graph_e)
            if sources == None:
                break
            source = [s for s in sources if nx.has_path(graph_e,s,i)][0] 
            if gen.bundle_value(source, i) + gen.get_val(i, g_i) > gen.bundle_value(i, i):
                bundle_j = gen.path_resolution(nx.shortest_path(graph_e, source, i)) # check if this is correct
                gen.assign_bundle_in_allocation(i, bundle_j)
                gen.add_good_from_pool(i, g_i)
                #gen.pool = gen.pool - {g_i}
                # gen.allocation[i] = bundle_j.append(g_i)
            
            else:
                gen.add_good_from_pool(source, g_i)
                # gen.pool = gen.pool - {g_i}
                # gen.allocation[source].append(g_i)
            new_graph_e = gen.get_enhanced_graph()
           
            gen.resolve_all_cycles(new_graph_e, type=type)

class AtMostSevenAllocate(Algorithm):
    def __init__(self, gen : GeneralFairAllocationInstance):
        self.N = gen.N
        self.M = gen.M
        self.G = gen
        T = ThreePA(self.G,plus=True)
        T.run()

    @collect_alg_timestamps_on_call
    def run(self):

        if self.N > 7:
            return
        # run the algorithm
        
        has_critical = self.G.get_critical_goods()
        sources = self.G.get_sources(self.G.get_enhanced_graph())
        C = [ g for  g in has_critical if len(has_critical[g]) > 1]
        
        if len(C) == 2 and len(sources) == 2:
            s_1, s_2 = sources[0], sources[1]
            g_1, g_2 = C[0], C[1]
            self.G.add_good_from_pool(s_1, g_1)
            self.G.add_good_from_pool(s_2, g_2)
            # self.G.allocation[s_1] = self.G.allocation[s_1].extend([g_1])
            # self.G.allocation[s_2] = self.G.allocation[s_2].extend([g_2])
            # self.G.pool = self.G.pool - {g_1, g_2}
        else:
            s = sources[0]
            for g in C:
                self.G.add_good_from_pool(s, g)

        # run the uncontested critical goods
        ThreePA.uncontested_critical(self.G, type='en')
        self.G.envy_cycle_elimination(self.G.get_enhanced_graph(), type='ev')

class MultigraphAllocate(Algorithm):
    def __init__(self, gen : MultiGraphInstance):
        self.N = gen.N
        self.M = gen.M
        self.G = gen
        T = ThreePA(self.G)
        T.run()
    
    # multigraph run 3PA
    @collect_alg_timestamps_on_call
    def run(self, type='en'):
        has_critical = self.G.get_critical_goods()
        sources = self.G.get_sources(self.G.get_enhanced_graph())
        C = [ g for  g in has_critical if len(has_critical[g]) > 1]
        s = sources[0]
        for g in C:
            self.G.add_good_from_pool(s, g)
        # self.G.allocation[s] = self.G.allocation[s].extend(C)
        ThreePA.uncontested_critical(self.G, type=type)
        self.G.envy_cycle_elimination(self.G.get_enhanced_graph())
    

        
class TopN(Algorithm):
    def __init__(self, gen : TopNInstance):
        self.N = gen.N
        self.M = gen.M
        self.G = gen
        self.G.gen_allocations(type='empty')
    

    @collect_alg_timestamps_on_call
    def run(self):
        # all agents agree on which are the top-N, but not necessarily on their value
        agents = self.G.agents
        T = self.G.get_top_n().copy()
        B = self.G.goods - T
        non_content = []
        for i in agents:
            if len(B) == 0 or len(T) == 0:
                break
            h_i = self.G.get_max_val_good(T, i)
            g_1i = min([(self.G.get_val(i,g), g) for g in T])[1]
            g_2i = self.G.get_max_val_good(B, i)
            if self.G.get_val(i, g_1i) + self.G.get_val(i, g_2i) >= (2/3)*self.G.get_val(i,h_i):
                self.G.add_good_from_pool(i, g_2i)
                non_content.append(i)
                B -= {g_2i}
            else:
                self.G.add_good_from_pool(i, h_i)
                T -= {h_i}

        for a in non_content:
            if len(T) == 0:
                break
            good = T.pop()
            self.G.add_good_from_pool(a, good)
            # self.G.allocation[a].append(good)
            # self.G.pool -= {good}
        self.G.envy_cycle_elimination(self.G.get_envy_graph(), type='ev')


class EnvyCycleElimination(Algorithm):
    def __init__(self, gen : GeneralFairAllocationInstance):
        self.N = gen.N
        self.M = gen.M
        self.G = gen
        self.G.gen_allocations(type='empty')
    
    @collect_alg_timestamps_on_call
    def run(self):
        env_G = self.G.get_envy_graph()
        self.G.envy_cycle_elimination(env_G, 'ev')

class RoundRobin(Algorithm):
    def __init__(self, gen : GeneralFairAllocationInstance):
        self.N = gen.N
        self.M = gen.M
        self.G = gen
        self.G.gen_allocations(type='empty')

    @collect_alg_timestamps_on_call
    def run(self):
        self.G.round_robin(list(self.G.agents))


if __name__ == '__main__':
    eps = 0.04
    x = np.array([
        [1,1,1,0,0,0,0,0.5+eps,0.5+eps],
        [1,1,1,0,0,0,0,0.5+eps,0.5+eps],
        [1,1,1,0,0,0,0,0.5+eps,0.5+eps],
        [1-2*eps,1-2*eps,1-2*eps,eps/2,eps/2,eps/2,eps/2,0,0]], dtype=np.float16)
    Gen = GeneralFairAllocationInstance(N=4, M=9, valuation_table=x)
    
    alg = ThreePA(Gen)
    alg.run()
    print(Gen.allocation)
    print(Gen.valuation_func)
    print(Gen.pool)

    plot_graphs([Gen.get_envy_graph(), Gen.get_enhanced_graph()])
    