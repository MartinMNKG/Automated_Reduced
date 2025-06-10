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
import sys 
import random 
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from GA.Tools import get_factor_dim_ln, rxns_yaml_arr_list2_ln,make_dir, write_yaml
from Database.Tools_0D import Sim0D,Processing_0D_ref,Processing_0D_data


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
    Restart : bool
    ) : 
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    main_path = os.getcwd()
    if rank == 0 & Restart == False  : # Create directory 
        dir = os.path.join(main_path,Name_Folder)
        os.makedirs(dir)
        make_dir(dir)


    Detailed_gas = ct.Solution(Detailed_file)
    Reduced_gas = ct.Solution(Reduced_file)


    if rank == 0 : 
        if Restart == False : # Launch simulation with Detailed mech 
            data_ref = Sim0D(Detailed_gas,Detailed_gas,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"Detailed","",False)
            Processing_Ref  = Processing_0D_ref(data_ref,cases_0D,length,"Detailed",dir,True) # Return all sim process into 1 datafram 
        else :  # Load Processing detailed simulation 
            Processing_Ref = pd.read_csv(os.path.join(dir,"Processing_Detailed.csv"))
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
        
        data = Sim0D(new_gas,new_gas,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"Individual","",False)
        Processing_Data = Processing_0D_data(data,Processing_Ref,cases_0D,"Individual","",False)
        
        Err =Fitness(Processing_Ref,Processing_Data,input_fitness,False)
        return Err, # Return tuple, Deap process 

    def mpi_evaluate(population):
        # Diviser la population entre les processus
        chunk_size = len(population) // size
        if rank == size - 1:
            chunk = population[rank * chunk_size:]  # Dernier processus prend le reste
        else:
            chunk = population[rank * chunk_size:(rank + 1) * chunk_size]
        
        # Évaluer le sous-ensemble assigné à ce processus
        local_results = list(map(toolbox.evaluate, chunk))
        
        # Rassembler les résultats dans le processus maître
        gathered_results = comm.gather(local_results, root=0)
        
        # Appliquer les résultats aux individus
        if rank == 0:
            flat_results = [item for sublist in gathered_results for item in sublist]
            for ind, fit in zip(population, flat_results):
                ind.fitness.values = fit
        comm.barrier()  # Synchronisation

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.Individual(create_individual(bounds)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend,alpha=0.5)
    toolbox.register("mutate", bounded_mutation, bounds=bounds, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("map", mpi_evaluate) 


    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", min)
    stats.register("avg",lambda fits: sum(fits) / len(fits))


    ## Main 
  
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals","min","avg"]

    if Restart == False : # Launch classic 
        ind_start = 1 
        population = toolbox.population(n=pop_size -1)
        special_inidivual = creator.Individual(init_value_factor) # Add Reduced as an individual
        population.append(special_inidivual)
        mpi_evaluate(population)

        comm.barrier()

        if rank == 0 : 
            record = stats.compile(population)
            logbook.record(gen=0, nevals=len(population), **record)
            
    else : # Take old population, and restart from 
        hist_path = os.path.join(dir,"hist")
        files = [f for f in os.listdir(hist_path) if re.match(r'population_\d+\.pkl', f)]
        files.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # Tri numérique
        last_population_file = files[-1]  # Prendre le dernier fichier
        with open(os.path.join(hist_path, last_population_file), "rb") as f:
                population = pickle.load(f)
                ind_start = int(re.search(r'\d+', last_population_file).group())

    for gen in range(ind_start,ngen +1) : 
        
        # Selection Best pop and keep it 
        if rank == 0 : 
            elite = tools.selBest(population, elitism_size)
            offspring = toolbox.select(population, len(population) - elitism_size )
            offspring = list(map(toolbox.clone, offspring))
        else : 
            offspring = None
        offspring = comm.bcast(offspring, root=0)
        
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
            
        # Evaluate New individual 
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        mpi_evaluate(invalid_ind)   
        
        comm.barrier()  
        
        # Concatenate Elite with new Individual 
        if rank == 0 : 
            population[:] = invalid_ind + elite
            with open(os.path.join(dir,"hist",f"population_{gen}.pkl"), "wb") as f:
                    pickle.dump(population, f) # Dump population if restart needed 
                    
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
            save_best = tools.selBest(population, 1)[0]
            opt_gas = rxns_yaml_arr_list2_ln(Reduced_gas,save_best)
            write_yaml(opt_gas ,os.path.join(dir,"mech",f"Mech_gen_{gen}.yaml"))
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

        generations = logbook.select("gen")
        min_fitness = logbook.select("min")
        avg_fitness = logbook.select("avg")
        plt.figure()
        plt.plot(generations, min_fitness, label="Min Fitness", marker='o')
        plt.plot(generations, avg_fitness, label="Avg Fitness", marker='x', linestyle='--')
        plt.title("Fitness Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dir,f"Fitness.png"))
        plt.figure() 
