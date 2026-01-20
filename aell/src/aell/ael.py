import numpy as np
import json
import random
from .ec.management import population_management
from .ec.interface_EC import InterfaceEC


# main class for AEL
class AEL:

    # initilization
    def __init__(self, use_local_llm, url, pop_size, n_pop, operators, m,
                 operator_weights, load_pop,
                 out_path, debug_mode, evaluation, ael_results_dir, **kwargs):

        # LLM settings
        self.use_local_llm = use_local_llm
        self.url = url
        self.evaluation = evaluation

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)

        assert isinstance(self.use_local_llm, bool)
        '''
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')
        # -------------------------------------------------------
        '''

        # Experimental settings
        self.pop_size = pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = n_pop  # number of populations

        self.operators = operators
        self.operator_weights = operator_weights
        if m > pop_size or m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            m = 2
        self.m = m

        self.debug_mode = debug_mode  # if debug
        self.ndelay = 1  # default
        self.load_pop = load_pop
        self.output_path = out_path
        self.ael_results_dir = ael_results_dir

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    # run ael
    def run(self, ob, object=False,pop_save_number=0):
        print("开始调用run函数")

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_eval = self.evaluation

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.debug_mode, interface_eval, use_local_llm=self.use_local_llm, url=self.url, object=object, ob=ob)

        # initialization
        population = []
        if 'use_seed' in self.load_pop and self.load_pop['use_seed']:
            import sys, os
            print(os.listdir())
            # os.chdir(r'D:\300work\303HKTasks\AEL\AEL-main\examples\QuadraticFunction')
            with open(self.load_pop['seed_path']) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data)
            filename = self.output_path + "/"+self.ael_results_dir+"/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop['use_pop']:  # load population from files
                print("load initial population from " + self.load_pop['pop_path'])
                with open(self.load_pop['pop_path']) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop['n_pop_initial']
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = population_management(population, self.pop_size)
                print("initial population has been created!")
                # Save population to a file
                filename = self.output_path + "/"+self.ael_results_dir+"/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        # main loop
        n_op = len(self.operators)

        for pop in range(n_start, self.n_pop):
            # Perform crossover and mutation
            for na in range(self.pop_size):
                for i in range(n_op):
                    op = self.operators[i]
                    op_w = self.operator_weights[i]
                    if (np.random.rand() < op_w):
                        parents, offspring = interface_ec.get_algorithm(population, op)
                    is_add = self.add2pop(population, offspring)  # Check duplication, and add the new offspring
                    print("generate new algorithm using " + op + " with fitness value: ", offspring['objective'])
                    if is_add:
                        data = {}
                        for i in range(len(parents)):
                            data[f"parent{i + 1}"] = parents[i]
                        data["offspring"] = offspring
                        with open(self.output_path + "/"+self.ael_results_dir+"/history/pop_" + str(pop + 1) + "_" + str(
                                na) + "_" + op + ".json", "w") as file:
                            json.dump(data, file, indent=5)

                # populatin management
                size_act = min(len(population), self.pop_size)
                population = population_management(population, size_act)
                print(f">> {na + 1} of {self.pop_size} finished ")

            print("fitness values of current population: ")
            for i in range(self.pop_size):
                print(str(population[i]['objective']) + " ")

            # Save population to a file
            filename = self.output_path + "/"+self.ael_results_dir+"/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file

            filename = self.output_path + "/"+self.ael_results_dir+"/pops_best/population_generation_" + str(pop + 1) + ".json"
            if pop_save_number==0:
                with open(filename, 'w') as f:
                    json.dump(population[0], f, indent=5)
            else:
                with open(filename, 'w') as f:
                    json.dump(population[:pop_save_number], f, indent=5)
            print(f">>> {pop + 1} of {self.n_pop} populations finished ")
        combined_results = {}
        for pop in range(n_start, self.n_pop):
            filename = os.path.join(self.output_path,
                                    self.ael_results_dir+"/pops_best/population_generation_" + str(pop + 1) + ".json")

            # 读取每个JSON文件
            if pop_save_number == 0:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 提取"code"和"objective"
                    combined_results[str(pop + 1)] = {
                        "code": data.get("code", ""),
                        "objective": data.get("objective", None),
                        "answer": data.get("answer", ""),
                        "parent1": data.get("parent1", ""),
                        "parent2": data.get("parent2", ""),
                    }
            else:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for pop, entry in enumerate(data):
                    combined_results[str(pop + 1)] = {
                        "code": entry.get("code", ""),
                        "objective": entry.get("objective", None),
                        "answer": entry.get("answer", ""),
                        "parent1": entry.get("parent1", ""),
                        "parent2": entry.get("parent2", ""),
                    }
        # 将结果写入新的JSON文件
        output_file_path = os.path.join(self.output_path, self.ael_results_dir, "combined_results.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=4)
        print("Results have been combined!")


        output_file_path = os.path.join(self.output_path, self.ael_results_dir+"/best_record", "combined_results0.json")
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=4)


    def find(self,epoch,ob, object=False,pop_size=None,n_pop=None,m=None,pop_save_number=0):
        print("开始调用find函数")
        if pop_size!=None:
            self.pop_size = pop_size
        if n_pop!=None:
            self.n_pop = n_pop
        if m!=None:
            self.m = m
        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_eval = self.evaluation 

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.debug_mode, interface_eval, use_local_llm=self.use_local_llm, url=self.url, object=object, ob=ob)

        import sys, os
        self.load_pop['seed_path']=os.path.join(self.output_path, self.ael_results_dir+"/best_record", "combined_results"+str(epoch)+".json")
        print("读取的种子路径为",self.load_pop['seed_path'])
        output_dir = os.path.dirname(self.load_pop['seed_path'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # initialization
        population = []
        if 'use_seed' in self.load_pop and self.load_pop['use_seed']:
            print('use_seed')
            import sys, os
            print(os.listdir())
            # os.chdir(r'D:\300work\303HKTasks\AEL\AEL-main\examples\QuadraticFunction')
            with open(self.load_pop['seed_path']) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed_find(data)
            filename = os.path.join(self.output_path, self.ael_results_dir, "best_record", "epoch_"+str(epoch + 1),"pops",
                                    "population_generation_0.json")
            output_dir = os.path.dirname(filename)
            print("")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop['use_pop']:  # load population from files
                print("load initial population from " + self.load_pop['pop_path'])
                with open(self.load_pop['pop_path']) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop['n_pop_initial']
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = population_management(population, self.pop_size)
                print("initial population has been created!")
                # Save population to a file
                filename = os.path.join(self.output_path, self.ael_results_dir, "best_record", "epoch_"+str(epoch + 1),"pops",
                                        "population_generation_0.json")
                output_dir = os.path.dirname(filename)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        # main loop
        n_op = len(self.operators)
        print("开始pop")

        for pop in range(n_start, self.n_pop):
            # Perform crossover and mutation
            for na in range(self.pop_size):
                for i in range(n_op):
                    op = self.operators[i]
                    op_w = self.operator_weights[i]
                    if (np.random.rand() < op_w):
                        parents, offspring = interface_ec.get_algorithm(population, op)
                    is_add = self.add2pop(population, offspring)  # Check duplication, and add the new offspring
                    print("generate new algorithm using " + op + " with fitness value: ", offspring['objective'])
                    if is_add:
                        data = {}
                        for i in range(len(parents)):
                            data[f"parent{i + 1}"] = parents[i]
                        data["offspring"] = offspring
                        filename = os.path.join(
                            self.output_path,
                            self.ael_results_dir,
                            "best_record",
                            "epoch_"+str(epoch + 1),
                            "history",
                            f"pop_{pop + 1}_{na}_{op}.json"
                        )
                        output_dir = os.path.dirname(filename)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        with open(filename, "w") as file:
                            json.dump(data, file, indent=5)

                # populatin management
                size_act = min(len(population), self.pop_size)
                population = population_management(population, size_act)
                print(f">> {na + 1} of {self.pop_size} finished ")

            print("fitness values of current population: ")
            for i in range(self.pop_size):
                print(str(population[i]['objective']) + " ")

            # Save population to a file
            filename = os.path.join(
                self.output_path,
                self.ael_results_dir,
                "best_record",
                "epoch_"+str(epoch + 1),
                "pops",
                f"population_generation_{pop + 1}.json"
            )
            output_dir = os.path.dirname(filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file
            filename = os.path.join(
                self.output_path,
                self.ael_results_dir,
                "best_record",
                "epoch_"+str(epoch + 1),
                "pops_best",
                f"population_generation_{pop + 1}.json"
            )
            output_dir = os.path.dirname(filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if pop_save_number == 0:
                with open(filename, 'w') as f:
                    json.dump(population[0], f, indent=5)
            else:
                with open(filename, 'w') as f:
                    json.dump(population[:pop_save_number], f, indent=5)
            print(f">>> {pop + 1} of {self.n_pop} populations finished ")
        combined_results = {}
        for pop in range(n_start, self.n_pop):
            filename = os.path.join(
                self.output_path,
                self.ael_results_dir,
                "best_record",
                "epoch_"+str(epoch + 1),
                "pops_best",
                f"population_generation_{pop + 1}.json"
            )
            output_dir = os.path.dirname(filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 读取每个JSON文件
            if pop_save_number == 0:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 提取"code"和"objective"
                    combined_results[str(pop + 1)] = {
                        "code": data.get("code", ""),
                        "objective": data.get("objective", None),
                        "answer": data.get("answer", ""),
                        "parent1": data.get("parent1", ""),
                        "parent2": data.get("parent2", ""),
                    }
            else:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for pop, entry in enumerate(data):
                    combined_results[str(pop + 1)] = {
                        "code": entry.get("code", ""),
                        "objective": entry.get("objective", None),
                        "answer": entry.get("answer", ""),
                        "parent1": entry.get("parent1", ""),
                        "parent2": entry.get("parent2", ""),
                    }
        # 将结果写入新的JSON文件
        output_file_path = os.path.join(self.output_path, self.ael_results_dir+"/best_record/combined_results"+ str(epoch+1) + ".json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=4)
        print("EPOCH " +str(epoch+1)+"!")