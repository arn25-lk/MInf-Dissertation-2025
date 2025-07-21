import math
import random
from advanced_algorithms import decorators
from  advanced_algorithms import advAlgorithms as alg

import traceback

from advanced_algorithms import group_champion
class Experiment:
    def run(self, data, func, prop=None):
        raise NotImplementedError("Subclasses should implement this!")

# Experiment 1
    # This experiment runs the algorithm on a set of instances
    # and collects the results. It can be used to test the performance
    # of the algorithm on different instances.

class RuntimeExperiment(Experiment):
    def run(self, data, func : alg.Algorithm,prop=None):
        try:
        # Example implementation of an experiment
            for instance in data:
                algorithm_instance = func(instance)
                result = algorithm_instance.run()
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error in instance: {instance.valuation_func, instance.allocation, instance.pool}") 
            traceback.print_exc()
            return False



# Experiment 2
# This experiment runs the algorithm on a set of instances
# and checks the runtime of the algorithm. It can be used to test
# the performance of the algorithm on different instances.
# The runtime can be used to compare the performance of different
# algorithms on the same instances.
class GroupChampionExperiment(Experiment):
    def run(self, data, func : alg.Algorithm, prop=None):
        result = []
        # Example implementation of an experiment
        for instance in data:
            # shuffle the agents in random order

                algorithm_instance = func(instance)
                algorithm_instance.run()
                G_champ, l_vertex = group_champion.get_group_champion_graph(instance)
                # create dictionary to store the group champion data like number of edges, number of vertices.  for each instance
                if G_champ.number_of_edges() > 0:
                    data = {
                        'num_agents': instance.N,
                        'num_goods': instance.M,
                        'instance': instance,
                        'num_edges': G_champ.number_of_edges(),
                        'num_vertices': len(G_champ.nodes),
                        'num_components': l_vertex,
                        'graph' : G_champ
                    }
                    result.append(data)
        return result


# Experiment 3
# This experiment runs the algorithm on a set of instances
# and checks if the algorithm satisfies a certain property.
# The property can be a function that takes an instance as input
# and returns a boolean value. If the property is not satisfied,
# the experiment will print the error and return False.
# This can be used to test the correctness of the algorithm
# and to check if it satisfies certain properties.
# Example properties could be: 
# - No agent envies another agent
# - The allocation is alpha-EFX
class CorrectnessTest(Experiment):
    def run(self, data, func : alg.Algorithm, prop=None):
        # Example implementation of an experiment
            for instance in data:
                try:
                    algorithm_instance = func(instance)
                    algorithm_instance.run()
                    assert prop(instance)
                except Exception as e:
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error in instance: {instance.valuation_func, instance.allocation, instance.pool}") 
                    traceback.print_exc()
                    return False

# Experiment 4
# This experiment runs on a set of instances and checks the number of critical goods and sources in the Enhanced Envy-Graph
# in a 3PA allocation
class CriticalGoodsAndSources(Experiment):
    def run(self, data : list[alg.GeneralFairAllocationInstance], func : alg.Algorithm, prop=None):
        results = []
        for instance in data:
            algorithm_instance = func(instance)
            algorithm_instance.run()
            # Collect the critical goods
            c = instance.get_critical_goods()
            v = instance.get_valuable_goods()
            G_e = instance.get_enhanced_graph()
            srcs = instance.get_sources(G_e)
            data = {}
            max_num_critical = 0
            max_num_valuable = 0
            if len(c) > 0:
                max_num_critical = max([len(c[i]) for i in c])
                print(c)
            if len(v) > 0:
                max_num_valuable = max([len(v[i]) for i in v])
            # Collect the data
            data['num_agents'] = instance.N
            data['num_goods'] = instance.M
            data['num_critical_goods'] = max_num_critical
            data['num_valuable_goods'] = max_num_valuable
            data['num_critical_goods'] = len(c)
            data['num_valuable_goods'] = len(v)
            data['num_sources'] = len(srcs)
            results.append(data)
        return results
    
# Experiment 5

class FrequencyExperiment(Experiment):
    def run(self, data : list[alg.GeneralFairAllocationInstance], func : alg.Algorithm, prop=None):
        results = []
        run = 1

        for instance in data:
            algorithm_instance = func(instance)
            algorithm_instance.run()
            # Collect the critical goods
            while not decorators.data_queue.empty():
                data = decorators.data_queue.get()
                results.append(data)
                run += 1
        return results