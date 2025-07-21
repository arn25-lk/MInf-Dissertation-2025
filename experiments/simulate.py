import enum
import random
import unittest
import threading
import time
import queue
from functools import wraps
from scipy.io import loadmat
import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from  advanced_algorithms import advAlgorithms as alg
from advanced_algorithms import indivGoodInstance as igi
from advanced_algorithms import decorators as dec
from advanced_algorithms import group_champion as gc

from experiments.experiment import GroupChampionExperiment, RuntimeExperiment, FrequencyExperiment, CriticalGoodsAndSources

class DataSetType(enum.Enum):
    STATIC = 1
    RANDOM = 2


class ValuationDistType(enum.Enum):
    QUADRATIC_FACE_DOWN = 1
    QUADRATIC_FACE_UP   = 2
    UNIFORM             = 3
    BINARY              = 4
    NORMAL              = 5
    UNIFORM_HOMOGENOUS  = 6

class InstanceType(enum.Enum):
    GENERAL    = 1 
    MULTIGRAPH = 2
    TOPN       = 3


class GeneralTestSuite:
    def __init__(self, data_set_type : DataSetType, 
                 experiment_class, instance_type : InstanceType , 
                 valuation_dist_type : ValuationDistType = None, 
                 static_source=None,num_instances=100, instance_size_bound=(16,50)):
        self.data_set_type = data_set_type
        self.experiment_class = experiment_class
        if data_set_type == DataSetType.STATIC:
            self.static_source = static_source
            self.num_instances = None
        if data_set_type == DataSetType.RANDOM:
            self.num_instances = num_instances
            self.valuation_dist_type = valuation_dist_type
        self.instance_type = self._returnClass(instance_type)
        self.instance_size_bound = instance_size_bound

    def _returnClass(self,case):
        match(case):
            case InstanceType.GENERAL:
                return igi.GeneralFairAllocationInstance
            case InstanceType.MULTIGRAPH:
                return igi.MultiGraphInstance
            case InstanceType.TOPN:
                return igi.TopNInstance
    
    def generate_data(self):
        if self.data_set_type == DataSetType.STATIC:
            if self.static_source == None:
                raise ValueError("Static Source Not Provided")
            data = loadmat(self.static_source)
            valuations = data.pop('valuations')[0]
            valuations = [np.array(v, dtype=np.uint32) for v in valuations]
            insts = [self.instance_type(N=v.shape[0], M=v.shape[1], valuation_table=v) for v in valuations if v.shape[0] <= self.instance_size_bound[0] and v.shape[1] <= self.instance_size_bound[1]]
            return insts
        
        elif self.data_set_type == DataSetType.RANDOM:
            dist_type = self.valuation_dist_type
            instances = []
            for _ in range(self.num_instances):
                # draw random N and M
                max_n, max_m = self.instance_size_bound
                N = random.randint(3, max_n)
                M = random.randint(2*N, max_m)

                instances.append(self.instance_type(N=N, M=M, dist=dist_type.name.lower(), params=[500, 250]))
            return instances
        else:
            raise ValueError("Unsupported data set type")

    def run_experiments(self, func : alg.Algorithm, params=None):
        data = self.generate_data()
        experiment = self.experiment_class()
    
        if params == None:
            result = experiment.run(data, func)
        else: 
            result = experiment.run(data, func, params)
        return result

            
    def process_timing_data(filename=None, folder_name=None):
        """Process and clear the data queue after each run.""" 
        os.makedirs("exp_data", exist_ok=True)
        if folder_name:
            os.makedirs(f"exp_data/{folder_name}", exist_ok=True)
        average_times = {}
        sum_times = {}
        while not dec.data_queue.empty():
            data = dec.data_queue.get()
            # Process the data as needed
            if data['function'] in average_times:
                average_times[data['function']] += data['exec_time']
                sum_times[data['function']] += 1
            else: 
                average_times[data['function']] = data['exec_time']
                sum_times[data['function']] = 1
        for key in average_times.keys():
            average_times[key] = average_times[key] / sum_times[key]


        

        with open(f"exp_data/{folder_name}/avg_time.txt", "a") as f:
            for key, value in average_times.items():
                f.write(f"{key}: {value}\n")
        f_name = filename if filename else "timing_data"
        folder_name = folder_name if folder_name else "timing"
   
        plt.savefig(f"exp_data/{folder_name}/{f_name}.png")
        plt.close()

        dec.data_queue.queue.clear()
    def process_critical_valuable_source(data, filename=None, folder_name=None):
        """
        Plots line graphs of:
        - critical goods per good
        - valuable goods per good
        - sources per good
        ordered by increasing number of goods.

        Parameters:
        - data: list of dictionaries with keys:
            'num_agents', 'num_goods', 'num_critical_goods', 'num_valuable_goods', 'num_sources'
        - filename: optional path to save the plot
        """
        os.makedirs("exp_data", exist_ok=True)
        if folder_name:
            os.makedirs(f"exp_data/{folder_name}", exist_ok=True)
        # Sort data by increasing num_goods
        # data_sorted = sorted(data, key=lambda d: (d['num_goods'] * d['num_agents'])/2)
        import seaborn as sns
        import pandas as pd
        from scipy.ndimage import gaussian_filter
        df = pd.DataFrame(data)

        # Avoid division by zero
        df = df[df['num_sources'] > 0].copy()
        
        # Compute ratios
        df['valuable_per_source'] = df['num_valuable_goods'] / df['num_sources']
        df['critical_per_source'] = df['num_critical_goods'] / df['num_sources']

        # Create pivot tables
        pivot_val = df.pivot_table(index='num_agents', columns='num_goods', values='valuable_per_source', aggfunc='mean')
        pivot_crit = df.pivot_table(index='num_agents', columns='num_goods', values='critical_per_source', aggfunc='mean')


        pivot_val_interp = pivot_val.interpolate(method='linear', axis=1)
        pivot_crit_interp = pivot_crit.interpolate(method='linear', axis=1)
        # Set up subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)


        sns.heatmap(pivot_val_interp, ax=axes[0], cmap='YlGnBu', linewidths=0.3, linecolor='gray')
        axes[0].set_title('Valuable Goods per Source')
        axes[0].set_xlabel('Number of Goods')
        axes[0].set_ylabel('Number of Agents')

        sns.heatmap(pivot_crit_interp, ax=axes[1], cmap='YlOrRd', linewidths=0.3, linecolor='gray')
        axes[1].set_title('Critical Goods per Source')
        axes[1].set_xlabel('Number of Goods')

        plt.suptitle('Goods per Source across Agent-Good Space', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # Save the plot
        f_name = filename if filename else "critical_valuable_source_data"
        folder_name = folder_name if folder_name else "cvs"
        if filename:
            plt.savefig(f'exp_data/{folder_name}/{f_name}')
        else:
            plt.show()


    def process_frequency_data(data, filename=None, folder_name=None):
        """
        Plots line graphs of:
        - critical goods per good
        - valuable goods per good
        - sources per good
        ordered by increasing number of goods.

        Parameters:
        - data: list of dictionaries with keys:
            'num_agents', 'num_goods', 'num_critical_goods', 'num_valuable_goods', 'num_sources'
        - filename: optional path to save the plot
        """
        os.makedirs("exp_data", exist_ok=True)
        if folder_name:
            os.makedirs(f"exp_data/{folder_name}", exist_ok=True)
        
        # Sort data by increasing num_goods
        # data_sorted = sorted(data, key=lambda d: (d['num_goods'] * d['num_agents'])/2)

        """
        Expects results to be a list of dictionaries, where each dictionary is the frequency of each step executed.
        Example: {"step1": count1, "step2": count2, ...}
        The plotting will respect the sequential execution property.
        """
        
        # Initialize a dictionary to store the aggregated frequencies for each step
        aggregated_frequencies = {f"step{i}": 0 for i in range(1, 9)}  # Adjust the number of steps as needed

        # Aggregate the counts across all runs
        for result in data:
            for step, count in result.items():
                aggregated_frequencies[step] += count

        # Calculate the frequency of each step considering the sequential property
        step_frequencies = {}

        # Calculate the differences for subsequent steps
        for i in range(1, 8):  # Assuming 8 steps; adjust as necessary
            step_frequencies[f"step{i}"] = aggregated_frequencies[f"step{i}"] - aggregated_frequencies[f"step{i+1}"]

    # The last step is calculated as the frequency of the last step
        step_frequencies["step8"] = aggregated_frequencies["step8"]  # For the last step, there is no next step to subtract

        # Prepare the data for plotting
        steps = list(step_frequencies.keys())
        frequencies = [step_frequencies[step] for step in steps]
        # find average frequency of each step as a dictionary 
        avg_frequencies = {step: freq / len(data) for step, freq in step_frequencies.items()}
        # save the frequencies to a file
        with open(f"exp_data/{folder_name}/avg_freq.txt", "a") as f:
            for key, value in avg_frequencies.items():
                f.write(f"{key}: {value}\n")
        # Plot the data as a bar chart
        # plt.figure(figsize=(10, 6))
        # plt.bar(steps, frequencies, color='lightcoral')
        # plt.xlabel('Steps')
        # plt.ylabel('Frequency of Execution')
        # plt.title('Frequency of Step Execution in Sequential Experiment')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # f_name = filename if filename else "frequency_data"
        # folder_name = folder_name if folder_name else "cvs"
        # if filename:
        #     plt.savefig(f'exp_data/{folder_name}/{f_name}')
        # else:
        #     plt.show()

    def process_group_champion_graphs(data, filename=None, folder_name=None):
        """
        Plots the group champion graph for each instance in the data.

        Parameters:
        - data: list of dictionaries with keys:
            'num_agents', 'num_goods', 'num_critical_goods', 'num_valuable_goods', 'num_sources'
        - filename: optional path to save the plot
        """
        os.makedirs("exp_data", exist_ok=True)
        if folder_name:
             os.makedirs(f"exp_data/{folder_name}", exist_ok=True)
        
        for i, result in enumerate(data):
            gen = result['instance']
            print(result['num_agents'], result['num_goods'], gen.allocation, gen.valuation_func, gen.pool)
            igi.plot_graphs([result['graph']], pos=gc.layout_by_component(result['graph']), save_path=f'exp_data/{folder_name}/{filename}_{i+1}.png')

            # plt.xlabel("Number of Agents")
            # plt.ylabel("Number of Edges")
            # plt.title("Agents vs. Edges in Champion Graphs")
        
            # plt.savefig(f'exp_data/{folder_name}/{filename}_{i+1}.png')
            # plt.close()

    def print_average_freq_data():
        from collections import defaultdict
        sums = defaultdict(float)
        counts = defaultdict(int)

        with open('exp_data/frequency_threepa/avg_freq.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    step, value = line.split(':')
                    step = step.strip()
                    value = float(value.strip())
                    sums[step] += value
                    counts[step] += 1
                except ValueError:
                    print(f"Skipping invalid line: {line}")
                    continue

        averages = {step: sums[step] / counts[step] for step in sums}
        return averages
    
    def print_average_timing_data():
        from collections import defaultdict
        sums = defaultdict(float)
        counts = defaultdict(int)

        with open('exp_data/timing_threepa/avg_time.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    step, value = line.split(':')
                    step = step.strip()
                    value = float(value.strip())
                    sums[step] += value
                    counts[step] += 1
                except ValueError:
                    print(f"Skipping invalid line: {line}")
                    continue

        averages = {step: sums[step] / counts[step] for step in sums}
        return averages

    
class GeneralRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up environment for tests."""
        os.environ["TESTING_MODE"] = "true"  # Enable decorators in testing

    def setUp(self):
        """Prepare the environment before each test."""
        dec.data_queue.queue.clear()  # Clear queue before each test

    def tearDown(self):
        """Clean up after each test."""
        dec.data_queue.queue.clear()

    @classmethod
    def tearDownClass(cls):
        """Reset environment after all tests."""
        os.environ["TESTING_MODE"] = "false"



    def test_runtime__atmost_static_data(self):
        os.environ["ALG_RUNTIME"] = "true"
        
        algs = [alg.AtMostSevenAllocate, alg.EnvyCycleElimination, alg.DraftAndEliminate, alg.RoundRobin]
        suites = [GeneralTestSuite(DataSetType.STATIC, RuntimeExperiment, InstanceType.GENERAL, static_source="spliddit_goods_data.mat") for _ in range(len(algs))]
        # create a new suite for each algorithm
        for i, a in enumerate(algs):
            result = suites[i].run_experiments(a)
        GeneralTestSuite.process_timing_data(filename='test_runtime_atmost7_spliddit_data', folder_name='timing_general')

    def test_runtime_atmost_random_data(self):
        os.environ["ALG_RUNTIME"] = "true"
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.AtMostSevenAllocate, alg.EnvyCycleElimination, alg.DraftAndEliminate, alg.RoundRobin]
            suites = [GeneralTestSuite(DataSetType.RANDOM, RuntimeExperiment, 
                                     InstanceType.GENERAL, val, num_instances=1000, instance_size_bound=(7,15)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
            GeneralTestSuite.process_timing_data(filename=f'runtime_atmost7_random_data_{val.name}', folder_name='timing_general')

    def test_runtime_multigraph_random_data(self):
        os.environ["ALG_RUNTIME"] = "true"
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.MultigraphAllocate, alg.EnvyCycleElimination, alg.DraftAndEliminate, alg.RoundRobin]
            suites = [GeneralTestSuite(DataSetType.RANDOM, RuntimeExperiment, 
                                     InstanceType.MULTIGRAPH, val, num_instances=1000, instance_size_bound=(25,51)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
            GeneralTestSuite.process_timing_data(filename=f'runtime_multigraph_random_data_{val.name}', folder_name='timing_general')

    def test_runtime_topn_random_data(self):
        os.environ["ALG_RUNTIME"] = "true"
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.TopN, alg.EnvyCycleElimination, alg.DraftAndEliminate, alg.RoundRobin, alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, RuntimeExperiment, 
                                     InstanceType.TOPN, val, num_instances=1000, instance_size_bound=(25,51)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
            GeneralTestSuite.process_timing_data(f'runtime_topn_random_data_{val.name}', folder_name='timing_general')

class Test3PARuntimeExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up environment for tests."""
        os.environ["TESTING_MODE"] = "true"  # Enable decorators in testing

    def setUp(self):
        """Prepare the environment before each test."""
        dec.data_queue.queue.clear()  # Clear queue before each test

    def tearDown(self):
        """Clean up after each test."""
        dec.data_queue.queue.clear()

    @classmethod
    def tearDownClass(cls):
        """Reset environment after all tests."""
        os.environ["TESTING_MODE"] = "false"


    def test_runtime_TOPN(self):
        os.environ["RUNTIME"] = "true"
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, RuntimeExperiment, 
                                     InstanceType.TOPN, val, num_instances=100, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
            GeneralTestSuite.process_timing_data(filename=f'threepa_runtime_topn_random_{val.name}', folder_name='timing_threepa')

    def test_runtime_MULTIGRAPH(self):
        os.environ["RUNTIME"] = "true"
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, RuntimeExperiment, 
                                     InstanceType.MULTIGRAPH, val, num_instances=100, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
            GeneralTestSuite.process_timing_data(filename=f'threepa_runtime_multigraph_random_{val.name}', folder_name='timing_threepa')


    def test_runtime_GENERAL(self):
        os.environ["RUNTIME"] = "true"
        
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, RuntimeExperiment, 
                                     InstanceType.GENERAL, val, num_instances=100, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
            GeneralTestSuite.process_timing_data(filename=f'threepa_runtime_general_random_{val.name}', folder_name='timing_threepa')

    def test_runtime_static_data(self):
        os.environ["RUNTIME"] = "true"
        suite = GeneralTestSuite(DataSetType.STATIC, RuntimeExperiment, InstanceType.GENERAL, static_source="spliddit_goods_data.mat")
        result = suite.run_experiments(alg.ThreePA)
        GeneralTestSuite.process_timing_data(filename=f'threepa_runtime_general_spliddit', folder_name='timing_threepa')

class Test3PACriticalValuableSourceExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up environment for tests."""
        os.environ["TESTING_MODE"] = "true"  # Enable decorators in testing

    def setUp(self):
        """Prepare the environment before each test."""
        dec.data_queue.queue.clear()  # Clear queue before each test

    def tearDown(self):
        """Clean up after each test."""
        dec.data_queue.queue.clear()

    @classmethod
    def tearDownClass(cls):
        """Reset environment after all tests."""
        os.environ["TESTING_MODE"] = "false"
    
    def test_static_critical_valuable_source(self):
        os.environ["RUNTIME"] = "true"
        algs = [alg.ThreePA]
        suites = [GeneralTestSuite(DataSetType.STATIC, CriticalGoodsAndSources, 
                                 InstanceType.GENERAL, static_source="spliddit_goods_data.mat") for _ in range(len(algs))]
        for i, a in enumerate(algs):
            result = suites[i].run_experiments(a)
            GeneralTestSuite.process_critical_valuable_source(result, filename='threepa_static_cvs', folder_name='cvs_threepa')
    
    def test_general_critical_valuable_source(self):
        os.environ["RUNTIME"] = "true"
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, CriticalGoodsAndSources, 
                                     InstanceType.GENERAL, val, num_instances=1000, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
                GeneralTestSuite.process_critical_valuable_source(result, filename=f'threepa_general_cvs_{val.name}',folder_name='cvs_threepa')

    def test_multigraph_critical_valuable_source(self):
        os.environ["RUNTIME"] = "true"
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, CriticalGoodsAndSources, 
                                    InstanceType.MULTIGRAPH, val, num_instances=1000, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
                GeneralTestSuite.process_critical_valuable_source(result, filename=f'threepa_multi_cvs_{val.name}', folder_name='cvs_threepa')

    def test_topn_critical_valuable_source(self):
        os.environ["RUNTIME"] = "true"
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, CriticalGoodsAndSources, 
                                    InstanceType.TOPN, val, num_instances=1000, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
                GeneralTestSuite.process_critical_valuable_source(result, filename=f'threepa_topn_cvs_{val.name}', folder_name='cvs_threepa')


class Test3PAFrequencyData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up environment for tests."""
        os.environ["TESTING_MODE"] = "true"  # Enable decorators in testing

    def setUp(self):
        """Prepare the environment before each test."""
        dec.data_queue.queue.clear()  # Clear queue before each test

    def tearDown(self):
        """Clean up after each test."""
        dec.data_queue.queue.clear()

    @classmethod
    def tearDownClass(cls):
        """Reset environment after all tests."""
        os.environ["TESTING_MODE"] = "false"
    
    def test_static_freq(self):
        os.environ["FREQUENCY"] = "true"
        self.setUp()
        algs = [alg.ThreePA]
        suites = [GeneralTestSuite(DataSetType.STATIC, FrequencyExperiment, 
                                 InstanceType.GENERAL, static_source="spliddit_goods_data.mat") for _ in range(len(algs))]
        for i, a in enumerate(algs):
            result = suites[i].run_experiments(a)
            GeneralTestSuite.process_frequency_data(result, filename='threepa_static_freq', folder_name='frequency_threepa')
    
    def test_general_freq(self):
        os.environ["FREQUENCY"] = "true"
        self.setUp()
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, FrequencyExperiment, 
                                     InstanceType.GENERAL, val, num_instances=250, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
                GeneralTestSuite.process_frequency_data(result, filename=f'threepa_general_freq_{val.name}',folder_name='frequency_threepa')

    def test_multigraph_freq(self):
        os.environ["FREQUENCY"] = "true"
        self.setUp()
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, FrequencyExperiment, 
                                    InstanceType.MULTIGRAPH, val, num_instances=250, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
                GeneralTestSuite.process_frequency_data(result, filename=f'threepa_multi_freq_{val.name}', folder_name='frequency_threepa')

    def test_topn_freq(self):
        os.environ["FREQUENCY"] = "true"
        self.setUp()
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, FrequencyExperiment, 
                                    InstanceType.TOPN, val, num_instances=250, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
                GeneralTestSuite.process_frequency_data(result, filename=f'threepa_topn_freq_{val.name}', folder_name='frequency_threepa')

    
class Test3PAGroupChampionGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up environment for tests."""
        os.environ["TESTING_MODE"] = "true"  # Enable decorators in testing

    def setUp(self):
        """Prepare the environment before each test."""
        dec.data_queue.queue.clear()  # Clear queue before each test

    def tearDown(self):
        """Clean up after each test."""
        dec.data_queue.queue.clear()

    @classmethod
    def tearDownClass(cls):
        """Reset environment after all tests."""
        os.environ["TESTING_MODE"] = "false"


    def test_general_group_champion(self):
        os.environ["FREQUENCY"] = "true"
        self.setUp()
        for val in [ValuationDistType.UNIFORM, ValuationDistType.BINARY, ValuationDistType.QUADRATIC_FACE_DOWN, ValuationDistType.QUADRATIC_FACE_UP, ValuationDistType.UNIFORM_HOMOGENOUS, ValuationDistType.NORMAL]:
            algs = [alg.ThreePA]
            suites = [GeneralTestSuite(DataSetType.RANDOM, GroupChampionExperiment, 
                                     InstanceType.GENERAL, val, num_instances=250, instance_size_bound=(25,100)) for _ in range(len(algs))]
            for i, a in enumerate(algs):
                result = suites[i].run_experiments(a)
                GeneralTestSuite.process_group_champion_graphs(result, filename=f'group_champ_general_{val.name}',folder_name='group_champ')

    def test_static_champion(self):
        os.environ["FREQUENCY"] = "true"
        self.setUp()

        algs = [alg.ThreePA]
        suites = [GeneralTestSuite(DataSetType.STATIC, GroupChampionExperiment,
                                    InstanceType.GENERAL, static_source="spliddit_goods_data.mat") for _ in range(len(algs))]
        for i, a in enumerate(algs):
            result = suites[i].run_experiments(a)
            GeneralTestSuite.process_group_champion_graphs(result, filename=f'group_champ_spliddit',folder_name='group_champ')
    
if __name__ == "__main__":
    # unittest.main()
    times= GeneralTestSuite.print_average_timing_data()
    freqs= GeneralTestSuite.print_average_freq_data()
    avg_values = {f'step{i}': (times[f'ThreePA.step{i}']/freqs[f'step{i}']) if freqs[f'step{i}'] != 0 else 0 for i in range(1, 9) }





    print(avg_values)
