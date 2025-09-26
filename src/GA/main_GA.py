from deap import base, creator, tools, algorithms
import numpy as np 
from mpi4py import MPI
import matplotlib.pyplot as plt

import pandas as pd 
import cantera as ct 
import os 
import re
import pickle
import csv 
import glob
import sys 
import time 
import random 
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from GA.Tools import get_factor_dim_ln, rxns_yaml_arr_list2_ln,make_dir, write_yaml
from Database.Tools_0D import Sim0D,Processing_0D_ref,Processing_0D_data
from Fitness.EEM import RE,RE_A,ABS,RMSE,IE


def Launch_GA(
    Name_Folder : str,
    Fitness,
    input_fitness,
    Detailed_file : str,
    Reduced_file : str,
    fuel1 :str,
    fuel2 :str,
    oxidizer : str, 
    tmax, 
    dt, 
    length : int,
    cases_0D, 
    pop_size : int, 
    ngen : int, 
    elitism_size : int, 
    cxpb,
    mutpb,
    type_fit : str,
    Restart : bool
    ) : 
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    main_path = os.getcwd()
    dir = Name_Folder
    if rank == 0 : 
        
        
        if Restart == False  : # Create directory 
            os.makedirs(dir,exist_ok=True)
            make_dir(dir)

             
    Detailed_gas = ct.Solution(Detailed_file)
    Reduced_gas = ct.Solution(Reduced_file)
    
    special_fit = (ABS,RE,RE_A,RMSE,IE) ### For input gestion only (EEM fitness type)


    if rank == 0 : 
        if Restart == False : # Launch simulation with Detailed mech 
            data_ref = Sim0D(Detailed_gas,Detailed_gas,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"","",False)
            data_start = Sim0D(Reduced_gas,Reduced_Gas,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"","",False)
            Processing_Ref  = Processing_0D_ref(data_ref,cases_0D,length,"Detailed",dir,True) # Return all sim process into 1 datafram 
            Processing_start = Processing_0D_data(data_start, Processing_Ref, cases_0D, "Start",dir, Ture)
            if Fitness in special_fit  : # Fit type EEM 
                list_spec = input_fitness.get("species")
                do_log = input_fitness.get("do_log")
                norm_type = input_fitess.get("norm_type")
                F_ref = Fitness(Processing_Ref,Processing_start,list_spec,False,do_log,norm_type)
            else : # Fit Type ORCH , PMO, Brookesia
                F_ref = Fitness(Processing_Ref, Processing_Data, input_fitness, False)
                
        else :  # Load Processing detailed simulation 
            Processing_Ref = pd.read_csv(os.path.join(dir,"Processing_Detailed.csv"))
            Processing_start = pd.read_csv(os.path.join(dir,"Processing_Start.csv"))
            if Fitness in special_fit  : # Fit type EEM 
                list_spec = input_fitness.get("species")
                do_log = input_fitness.get("do_log")
                norm_type = input_fitess.get("norm_type")
                F_ref = Fitness(Processing_Ref,Processing_start,list_spec,False,do_log,norm_type)
            else : # Fit Type ORCH , PMO, Brookesia
                F_ref = Fitness(Processing_Ref, Processing_Data, input_fitness, False)
    else :  
        data_ref = None 
        Processing_Ref = None 
    comm.barrier() 
    Processing_Ref = comm.bcast(Processing_Ref,root=0)   

    variation_percent = 0.1
    num_individu,init_value_factor = get_factor_dim_ln(Reduced_gas) # Get A , B , E from mech 
    bounds = [(val * (1 - variation_percent), val * (1 + variation_percent)) for val in init_value_factor] # Create bounds +- 10 % 



    def create_gene(lower, upper):
            return random.uniform(lower, upper)
    def create_individual(bounds):
        return [create_gene(lower, upper) for lower, upper in bounds]

    def bounded_mutation(individual, bounds, mu, sigma, indpb):
        # Apply Gaussian mutation
        tools.mutGaussian(individual, mu, sigma, indpb)
        # Enforce bounds
        for i, (lower, upper) in enumerate(bounds):
            individual[i] = max(min(individual[i], upper), lower)
        return individual

    def repair(individual):
        """Repairs individual values to ensure they stay within bounds."""
        for i, (lower, upper) in enumerate(bounds):
            individual[i] = max(min(individual[i], upper), lower)
        return individual

    def evaluate(individual): 
        new_gas = rxns_yaml_arr_list2_ln(Reduced_gas,individual)
            
        data = Sim0D(new_gas,new_gas,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"","",False)
        
        Processing_Data = Processing_0D_data(data,Processing_Ref,cases_0D,f"","",False)
        Err =Fitness(Processing_Ref,Processing_Data,input_fitness,False)
        return Err, # Return tuple, Deap process 
    
    def evaluate_with_context(individual, gen, ind_index):
        start_time_evaluate = time.time()
        new_gas = rxns_yaml_arr_list2_ln(Reduced_gas, individual)
    

        data = Sim0D(new_gas, new_gas, fuel1, fuel2, oxidizer, cases_0D, dt, tmax, "", "", False)
      
        Processing_Data = Processing_0D_data(data, Processing_Ref, cases_0D, "", "", False)
        
        
        
        if Fitness in special_fit  : # Fit type EEM 
            list_spec = input_fitness.get("species")
            do_log = input_fitness.get("do_log")
            norm_type = input_fitess.get("norm_type")
            
            Err = Fitness(Processing_Ref,Processin_Data,list_spec,False,do_log,norm_type)
            
        else : # Fit Type ORCH , PMO, Brookesia
            Err = Fitness(Processing_Ref, Processing_Data, input_fitness, False)
      
        return Err,
    

    def mpi_evaluate(population, gen):
        # Répartition round-robin des indices
        indices = list(range(len(population)))
        chunks = [indices[i::size] for i in range(size)]  # round-robin

        # Chaque rank récupère ses indices + individus associés
        local_indices = chunks[rank]
        # print(f"RANK{rank},CHUNK : {chunks}")
    
        local_individuals = [population[i] for i in local_indices]

    

        local_results = [evaluate_with_context(ind, gen, i) for ind, i in zip(local_individuals, local_indices)]

        gathered_results = comm.gather((local_indices, local_results), root=0)

        if rank == 0:
            for inds, fits in gathered_results:
                for i, fit in zip(inds, fits):
                    population[i].fitness.values = fit
        comm.barrier()

        
    if type_fit == "Mini" :
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
    if type_fit == "Maxi" :
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.Individual(create_individual(bounds)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_with_context)
    toolbox.register("mate", tools.cxBlend,alpha=0.5)
    toolbox.register("mutate", bounded_mutation, bounds=bounds, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("map", mpi_evaluate) 


    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    if type_fit =="Mini" : 
        stats.register("min", min)
    if type_fit == "Maxi" : 
        stats.register("max", max)
    stats.register("avg",lambda fits: sum(fits) / len(fits))


    ## Main 
  
    logbook = tools.Logbook()
    if type_fit =="Mini" : 
        logbook.header = ["gen", "nevals","min","avg"]
    if type_fit == "Maxi" : 
        logbook.header = ["gen", "nevals","max","avg"]
    

    if Restart == False : # Launch classic 
        ind_start = 1 
        population = toolbox.population(n=pop_size -1)
        special_inidivual = creator.Individual(init_value_factor) # Add Reduced as an individual
        population.append(special_inidivual)
        
        
        with open(os.path.join(dir,"hist",f"population_0.pkl"), "wb") as f:
            pickle.dump(population, f) # Dump population if restart needed 
        mpi_evaluate(population,0)

        comm.barrier()

        if rank == 0 : 
            record = stats.compile(population)
            logbook.record(gen=0, nevals=len(population), **record)
            fitness_file = os.path.join(dir, "hist", f"fitness_gen_0.txt")
            with open(fitness_file, "w") as f_fit:
                for ind in population:
                    f_fit.write(f"{ind.fitness.values}\n") # Save fitness of pop at a each gen 
            
    else : # Take old population, and restart from 
        hist_path = os.path.join(dir,"hist")
        files = [f for f in os.listdir(hist_path) if re.match(r'population_\d+\.pkl', f)]
        files.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # Tri numérique
        last_population_file = files[-1]  # Prendre le dernier fichier
        with open(os.path.join(hist_path, last_population_file), "rb") as f:
                population = pickle.load(f)
                ind_start = int(re.search(r'\d+', last_population_file).group())
                
    for gen in range(ind_start,ngen +1) : 
        start_time =time.time() 
        # Selection Best pop and keep it 
        if rank == 0 : 
            
            
            # elite = tools.selBest(population, elitism_size)
            elite = list(map(toolbox.clone, tools.selBest(population, elitism_size)))
            offspring = toolbox.select(population, len(population) - elitism_size )
            offspring = list(map(toolbox.clone, offspring))
                
            # Create child 
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values      
                
            # Mutation Process 
            for mutant in offspring:

                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                repair(mutant)
            
        else : 
            offspring = None 
        offspring= comm.bcast(offspring,root=0)
            
        # Evaluate New individual 
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        mpi_evaluate(offspring,gen)   
        
        
        comm.barrier()  
        
        # Concatenate Elite with new Individual 
        if rank == 0 : 
            print(f"Time End GA : {time.time() - start_time}")
            population[:] = offspring + elite
            
    
            with open(os.path.join(dir,"hist",f"population_{gen}.pkl"), "wb") as f:
                    pickle.dump(population, f) # Dump population if restart needed 
                                    
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(offspring), **record)
            print(logbook.stream)
            
            save_best = tools.selBest(population, 1)[0]
            best_mech = rxns_yaml_arr_list2_ln(Reduced_gas,save_best)
            write_yaml(best_mech ,os.path.join(dir,"mech",f"Mech_gen_{gen}.yaml"))
            
            with open(os.path.join(dir, "hist", f"Output_mpi_gen{gen}.csv"), "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(logbook.header)  # Write the header (columns)
                for record in logbook:
                    writer.writerow(record.values())  # Write each generation's data
    
    if rank == 0 :    # End GA 
        best_individual = tools.selBest(population, k=1)[0]
        print("Best fitness:", best_individual.fitness.values[0])
        opt_gas = rxns_yaml_arr_list2_ln(Reduced_gas,best_individual)
        data = Sim0D(opt_gas,opt_gas,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"Individual","",False)
        Processing_Data = Processing_0D_data(data,Processing_Ref,cases_0D,"Optimized",dir,True)
        
        
        write_yaml(opt_gas ,os.path.join(dir,f"/Best_Individual.yaml"))
        opt_gas.write_yaml(os.path.join(dir,f"/Best_Individual_canteraapi.yaml"))

        generations = logbook.select("gen")
        min_fitness = logbook.select("min")
        avg_fitness = logbook.select("avg")
        plt.figure()
        plt.plot(generations, min_fitness, label="Min Fitness", marker='o')
        plt.yscale("log")
        plt.title("Fitness Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dir,f"Fitness.png"))
        
        plt.figure()
        plt.plot(generations, min_fitness/F_ref, label=f"Min Fitness,F_ref = {F_ref}", marker='o')
        plt.yscale("log")
        plt.title("Fitness Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dir,f"F_FREF.png"))
        plt.figure()
         
