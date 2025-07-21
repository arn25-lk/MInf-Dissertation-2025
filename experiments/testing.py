import math
import unittest
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import advanced_algorithms.advAlgorithms as advAlgorithms
import advanced_algorithms.indivGoodInstance as igi
import random 
import experiments.experiment as exp
import experiments.simulate as sim
import networkx as nx



class TestIndivGoodInstance(unittest.TestCase):
    # def test_gen_allocations(self):
    #     table = [[1, 3, 2, 5, 7, 4, 6, 8], 
    #          [1, 3, 2, 5, 7, 4, 6, 8], 
    #          [1, 3, 2, 5, 7, 4, 6, 0], 
    #          [1, 3, 2, 5, 0, 4, 6, 0], 
    #          [1, 3, 2, 5, 7, 4, 6, 8]]
    #     G = igi.GeneralFairAllocationInstance(N=5, M=8, valuation_table=table)
    #     G.gen_allocations(type='empty')
    #     self.assertEqual(G.pool,G.goods)
    #     G.gen_allocations(type='random')
    #     self.assertEqual(G.pool, {})
    #     G.gen_allocations(type='random_partial')
    #     s = []
    #     for a, gs in G.allocation.items():
    #         s.extend(gs)
    #     self.assertEqual(G.pool, G.goods - set(s))

    # def test_round_robin(self):
    #     table = [[1, 3, 2, 5, 7, 4, 6, 8], 
    #          [1, 3, 2, 5, 7, 4, 6, 8], 
    #          [1, 3, 2, 5, 7, 4, 6, 0], 
    #          [1, 3, 2, 5, 0, 4, 6, 0], 
    #          [1, 3, 2, 5, 7, 4, 6, 8]]
    #     G = igi.GeneralFairAllocationInstance(N=5, M=8, valuation_table=table)
    #     G.gen_allocations(type='empty')
    #     print(f'Before RR: {G.allocation}')
    #     G.round_robin()
    #     print(f'After RR: {G.allocation}')
    #     # ====
    #     G.gen_allocations(type='empty')
    #     print(f'Before RR: {G.allocation}')
    #     G.round_robin(max_iter=1)
    #     print(f'After RR: {G.allocation}')
    #     # ====
    #     G.gen_allocations(type='empty')
    #     print(f'Before RR: {G.allocation}')
    #     agents = list(G.agents)
    #     G.round_robin(order=agents[::-1], max_iter=1)
    #     print(f'After RR: {G.allocation}')
    #     # ==== 
    #     G.gen_allocations(type='random_partial')
    #     print(f'Before RR: {G.allocation}')
    #     G.round_robin()
    #     print(f'After RR: {G.allocation}')

    def test_graphs(self):
        table = [[1, 9, 2, 5, 7, 4, 6, 8], 
             [1, 3, 2, 5, 7, 4, 6, 8], 
             [1, 3, 2, 5, 7, 4, 6, 0], 
             [1, 3, 2, 5, 0, 4, 6, 0], 
             [1, 3, 2, 0, 0, 4, 0, 0]]
        G = igi.GeneralFairAllocationInstance(N=5, M=8, valuation_table=table)
        d = {1: [8], 2: [5], 3:[7], 4:[6,1], 5:[2]}
        G.gen_allocations(allocation_dict=d, type='from_dict')
        G_r = G.get_reduced_graph(debug=True)
        G_e = G.get_enhanced_graph(debug=True)
        Gx = G.get_envy_graph(debug=True)
        self.assertTrue(G.contains_cycles(Gx))
        self.assertFalse(G.contains_cycles(G_r))

        igi.plot_graphs([Gx, G_r], labels=["Envy", "Reduced"])
        

    # def test_envy_cycle_elim(self):
    #     table = [[1, 9, 2, 5, 7, 4, 6, 8, 7, 3, 7, 2], 
    #          [1, 3, 2, 5, 7, 4, 6, 8, 7, 3, 7, 2], 
    #          [1, 3, 2, 5, 7, 4, 6, 0, 7, 3, 7, 2], 
    #          [1, 3, 2, 5, 0, 4, 6, 0, 7, 3, 7, 2], 
    #          [1, 3, 2, 0, 6, 4, 0, 0, 7, 3, 7, 2]]
    #     G = igi.GeneralFairAllocationInstance(N=5, M=12, valuation_table=table)
    #     d = {1: [8], 2: [5], 3:[7], 4:[6,1], 5:[2]}
    #     G.gen_allocations(allocation_dict=d, type='from_dict')
    #     graphs = G.envy_cycle_elimination(G.get_envy_graph(), return_graphs=True)
    #     self.assertEqual(G.pool, set())
    #     graph = G.get_envy_graph()
    #     self.assertFalse(G.contains_cycles(graph))
    #     igi.plot_graphs(graphs)

    #     # random partial
    #     table, n, m  = random_valuation_table()
    #     G = indivGoodInstance.GeneralFairAllocationInstance(N=n, M=m, valuation_table=table)
    #     G.gen_allocations(type='empty') 
    #     graphs = G.envy_cycle_elimination(G.get_envy_graph(), return_graphs=True)
    #     self.assertEqual(G.pool, set())
    #     #self.assertFalse(G.contains_cycles(graph))


    # def test_resolve_cycle(self):
    #     table = [[1, 9, 2, 5, 7, 4, 6, 8], 
    #          [1, 3, 2, 5, 7, 4, 6, 8], 
    #          [1, 3, 2, 5, 7, 4, 6, 0], 
    #          [1, 3, 2, 5, 0, 4, 6, 0], 
    #          [1, 3, 2, 0, 0, 4, 0, 0]]
    #     G = igi.GeneralFairAllocationInstance(N=5, M=12, valuation_table=table)
    #     d = {1: [8], 2: [5], 3:[7], 4:[6,1], 5:[2]}
    #     G.gen_allocations(allocation_dict=d, type='from_dict')
        
    #     graph = G.get_envy_graph()
    #     cycle = nx.find_cycle(graph, orientation='original')
    #     G.resolve_cycle(cycle)
    #     self.assertEqual(G.allocation, {1: [2], 2: [8], 3: [5], 4: [7], 5: [6,1]})
    #     igi.plot_graphs([graph, G.get_envy_graph()])
    #     # ======
    #     table = [[1, 9, 2, 5, 7, 4, 6, 8, 7, 3, 7, 2], 
    #          [1, 3, 2, 5, 7, 4, 6, 8, 7, 3, 7, 2], 
    #          [1, 3, 2, 5, 7, 4, 6, 0, 7, 3, 7, 2], 
    #          [1, 3, 2, 5, 0, 4, 6, 0, 7, 3, 7, 2], 
    #          [1, 3, 2, 0, 6, 4, 0, 0, 7, 3, 7, 2]]
    #     G = igi.GeneralFairAllocationInstance(N=5, M=12, valuation_table=table)
    #     d = {1: [8], 2: [5], 3:[7], 4:[6,1], 5:[2]}
    #     G.gen_allocations(allocation_dict=d, type='from_dict')
    #     graph = G.get_envy_graph()
    #     cycle = nx.find_cycle(graph, orientation='original')
    #     G.resolve_cycle(cycle)
    #     self.assertEqual(G.allocation, {1: [2], 2: [8], 3: [7], 4: [6, 1], 5: [5]})
        
    #     igi.plot_graphs([graph, G.get_envy_graph()])

    def test_path_resolution(self):
        table = [[1, 9, 2, 5, 7, 4, 6, 8], 
             [1, 3, 2, 5, 7, 4, 6, 8], 
             [1, 3, 2, 5, 7, 4, 6, 0], 
             [1, 3, 2, 5, 0, 4, 6, 0], 
             [1, 3, 2, 0, 0, 4, 0, 0]]

        G = igi.GeneralFairAllocationInstance(N=5, M=8, valuation_table=table)
        d = {1: [8], 2: [5], 3:[7], 4:[6,1], 5:[2]}
        G.gen_allocations(allocation_dict=d, type='from_dict')

        graph = G.get_enhanced_graph()
        print(f"Before : {G.allocation}")
        paths = nx.all_simple_paths(graph, source=3, target=1)
        for p in paths:
            G.path_resolution_star(p, 1)
            break
        print(f"After : {G.allocation}")
        igi.plot_graphs([graph, G.get_enhanced_graph()],labels=["Before", "After"])      
  


class TestCorrectnessAdvAlgorithms(unittest.TestCase):

    def test_draft_and_eliminate(self):
        print("Running Draft-and-Eliminate tests...")
        for e in [sim.ValuationDistType.BINARY, sim.ValuationDistType.UNIFORM_HOMOGENOUS, 
                  sim.ValuationDistType.UNIFORM, sim.ValuationDistType.QUADRATIC_FACE_DOWN, sim.ValuationDistType.QUADRATIC_FACE_UP ]:
            A = advAlgorithms.DraftAndEliminate
            suite = sim.GeneralTestSuite(sim.DataSetType.RANDOM, exp.CorrectnessTest, sim.InstanceType.GENERAL, valuation_dist_type=e, num_instances=250)
            suite.run_experiments(A, params=verify_phi_efx)

    def test_three_PA(self):
        print("Running ThreePA tests...")
        for e in [sim.ValuationDistType.BINARY, sim.ValuationDistType.UNIFORM_HOMOGENOUS, 
                  sim.ValuationDistType.UNIFORM, sim.ValuationDistType.QUADRATIC_FACE_DOWN, sim.ValuationDistType.QUADRATIC_FACE_UP]:
            A = advAlgorithms.ThreePA 
            suite = sim.GeneralTestSuite(sim.DataSetType.RANDOM, exp.CorrectnessTest, sim.InstanceType.GENERAL, valuation_dist_type=e, num_instances=100)
            suite.run_experiments(A, params=verify_threePA)

    

    def test_static_three_PA(self):
        print("Running Static ThreePA tests...")
        A = advAlgorithms.ThreePA 
        suite = sim.GeneralTestSuite(sim.DataSetType.STATIC, exp.CorrectnessTest, sim.InstanceType.GENERAL, static_source="spliddit_goods_data.mat")
        suite.run_experiments(A, params=verify_threePA)

    def test_ThreePA_top_n(self):
        for e in [sim.ValuationDistType.UNIFORM, sim.ValuationDistType.QUADRATIC_FACE_DOWN, 
                  sim.ValuationDistType.QUADRATIC_FACE_UP, sim.ValuationDistType.BINARY, sim.ValuationDistType.UNIFORM_HOMOGENOUS]:
            A = advAlgorithms.TopN
            suite = sim.GeneralTestSuite(sim.DataSetType.RANDOM, exp.CorrectnessTest, sim.InstanceType.TOPN, valuation_dist_type=e, num_instances=250)
            suite.run_experiments(A, params=verify_alpha_efx)

    # def test_multigraph_allocate(self):
    #     for e in [sim.ValuationDistType.UNIFORM, sim.ValuationDistType.QUADRATIC_FACE_DOWN, sim.ValuationDistType.QUADRATIC_FACE_UP ]:
    #         A = advAlgorithms.MultigraphAllocate
    #         suite = sim.GeneralTestSuite(sim.DataSetType.RANDOM, exp.CorrectnessTest, sim.InstanceType.GENERAL, valuation_dist_type=e, num_instances=250)
    #         suite.run_experiments(A, params=verify_multigraph)
    def test_top_n(self):
        for e in [sim.ValuationDistType.UNIFORM, sim.ValuationDistType.QUADRATIC_FACE_DOWN, 
                  sim.ValuationDistType.QUADRATIC_FACE_UP, sim.ValuationDistType.BINARY, sim.ValuationDistType.UNIFORM_HOMOGENOUS]:
            A = advAlgorithms.TopN
            suite = sim.GeneralTestSuite(sim.DataSetType.RANDOM, exp.CorrectnessTest, sim.InstanceType.TOPN, valuation_dist_type=e, num_instances=250)
            suite.run_experiments(A, params=verify_alpha_efx)
    def test_seven_agent(self):
        for e in [sim.ValuationDistType.UNIFORM, sim.ValuationDistType.QUADRATIC_FACE_DOWN, 
                  sim.ValuationDistType.QUADRATIC_FACE_UP, sim.ValuationDistType.BINARY, sim.ValuationDistType.UNIFORM_HOMOGENOUS]:
            A = advAlgorithms.AtMostSevenAllocate
            suite = sim.GeneralTestSuite(sim.DataSetType.RANDOM, exp.CorrectnessTest, sim.InstanceType.TOPN, valuation_dist_type=e, num_instances=250)
            suite.run_experiments(A, params=verify_alpha_efx)

def verify_phi_efx(indivGood : igi.GeneralFairAllocationInstance):
    phi = ( 1 + math.sqrt(5) ) / 2
    return verify_alpha_efx(indivGood, alpha=(phi-1))

def verify_alpha_efx( indivGood : igi.GeneralFairAllocationInstance, alpha=2/3, agent=None):
    agents = indivGood.agents
    if agent:
        agents = [agent]
    
    for i in agents:
        
        vi_Ai = indivGood.bundle_value(i,i)
        for j in agents:
            if i == j:
                continue 
            vi_Aj = indivGood.bundle_value(j,i)
            for g in indivGood.allocation[j]:
                if vi_Ai < alpha * (vi_Aj - indivGood.get_val(i,g)):
                    print(i, j, g)
                    return False
    return True

def critical_goods(ind : igi.GeneralFairAllocationInstance):
    agents = ind.agents
    pool = ind.pool
    has_critical = {}
    for g in pool:
        for i in agents:
            vi_Ai = ind.bundle_value(i,i)
            vi_g = ind.get_val(i,g)
            # Skip cases where both are zero
            if vi_Ai < 2*vi_g:
                if i in has_critical:
                    has_critical[i].append(g)
                else: 
                    has_critical[i] = [g]
    return has_critical

def verify_multigraph(ind : igi.GeneralFairAllocationInstance):
    crit_goods = critical_goods(ind)
    assert len(crit_goods) == 0
    assert verify_alpha_efx(ind, alpha=2/3)
    return True

def verify_threePA(ind : igi.GeneralFairAllocationInstance):
    agents = ind.agents
    crit_goods = critical_goods(ind)
    for i in agents:
        if len(ind.allocation[i]) == 1:
            assert verify_alpha_efx(ind, alpha=1, agent=i)
            if i in crit_goods:
                assert len(crit_goods[i]) == 1 
        if len(ind.allocation[i]) == 2:
            assert verify_alpha_efx(ind, alpha=2/3, agent=i)
            assert i not in crit_goods
    return True


# Look at a critical good and see what happens
if __name__ == "__main__":
    unittest.main()