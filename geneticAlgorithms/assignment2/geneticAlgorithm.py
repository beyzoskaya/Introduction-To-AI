

import numpy as np
from dataset_reader import requirements, proficiency_levels
import random


# Getting parameters from the reader
number_of_students = requirements.shape[0]
number_of_projects = requirements.shape[1]
number_of_skills = proficiency_levels.shape[1]
population_size = 50


# print(f"requirements for projects {requirements}")
# print(f"shape for requirements {requirements.shape}")
# print(f"proficiency levels {proficiency_levels}")
# print(f" shape of proficiency levels {proficiency_levels.shape}")

#best elitism rate for population
elitism_rate = 0.1
#appropriate mutation rate
mutation_rate = 0.001
#selection rate for normal selection
selection_rate = 0.25
#what percentage of the population include tournament selection
tournament_size_percentage = 0.1


#RANDOM INITIALIZATON
def initiate_population(population_size, number_of_students, number_of_projects):
    population = []

    #not any student assign to any project at first
    for _ in range(population_size):
        chromosome = np.zeros((number_of_students, number_of_projects))

    
        for j in range(number_of_projects):

            #Random assignment for at most three at least 1 student to the projects
            num_assigned_students = np.random.randint(1, 4)
            assigned_students = np.random.choice(number_of_students, size=num_assigned_students, replace=False)
            #binary representation for assigned student to the j'th project
            chromosome[assigned_students, j] = 1
        #students keep as a chromosome so keep them in a list named population for after return it
        population.append(chromosome)

    
    for i, chromosome in enumerate(population):
        print(f"Chromosome {i}:")
        print(chromosome)
        print()

    return population
#re-initialize population for another iteration, same function iterative way
population = initiate_population(population_size,number_of_students,number_of_projects)


#RANDOM INITIALIZATION WITH ONE MORE CONSTRAINT
#updated random initiation for at most 2 projects can be assigned to the same project
# def initiate_population(population_size, number_of_students, number_of_projects):

#     population = []

#again firstly all the students not assigned to the any project
#     for _ in range(population_size):
#         chromosome = np.zeros((number_of_students, number_of_projects))

#         #Assign students to project again random way
#         for j in range(number_of_projects):
#             num_assigned_students = np.random.randint(0, 3)  
#             if num_assigned_students > 0:
#                 assigned_students = np.random.choice(number_of_students, size=num_assigned_students, replace=False)
                  #binary representation for assigned student
#                 chromosome[assigned_students, j] = 1

#         #keep the number of assigned students
#         while np.sum(chromosome, axis=1).max() > 2:
#             #random choice for more than 2 assignments
#             student_indices = np.where(np.sum(chromosome, axis=1) > 2)[0]
#             student = np.random.choice(student_indices)
            
#            #specific projects that are assigned to the random selected student
#             assigned_projects = np.where(chromosome[student] == 1)[0]
            
#             #select a project randomly for remove 
#             project_to_unassign = np.random.choice(assigned_projects)
            
#            #binary representation for selected project is not assigned to the student
#             chromosome[student, project_to_unassign] = 0
          #keep the chromosome for population
#         population.append(chromosome)

#    
#     for i, chromosome in enumerate(population):
#         print(f"Chromosome {i}:")
#         print(chromosome)
#         print()

#     return population
#re-initialize population
# population = initiate_population(population_size,number_of_students,number_of_projects)




#GREEDY INITIALIZATION FOR POPULATION
# def greedy_initiate_population(population_size, number_of_students, number_of_projects):
#     population = []

#     for _ in range(population_size):
#         chromosome = np.zeros((number_of_students, number_of_projects))

#         remaining_projects = list(range(number_of_projects))

#         for student in range(number_of_students):
#             if remaining_projects:
#                 # Sort projects based on their requirements and proficiency levels
#                 sorted_projects = sorted(remaining_projects, key=lambda project: np.sum(proficiency_levels[student] * requirements[:, project][:, None]))

#                 # Assign the student to the project with the highest requirement and proficiency level
#                 best_project = sorted_projects[-1]
#                 chromosome[student, best_project] = 1

#                 # Remove the assigned project from the remaining projects
#                 remaining_projects.remove(best_project)
#             else:
#                 # If there are no remaining projects, randomly assign the student to an available project
#                 available_projects = np.where(np.sum(chromosome, axis=0) == 0)[0]
#                 if available_projects.size > 0:
#                     random_project = random.choice(available_projects)
#                     chromosome[student, random_project] = 1

#         population.append(chromosome)

#     # Print the chromosomes and the population after the loop
#     for i, chromosome in enumerate(population):
#         print(f"Chromosome {i}:")
#         print(chromosome)
#         print()

#     return population


# population = greedy_initiate_population(population_size,number_of_students,number_of_projects)



#FITNESS FUNCTION CALCULATION FOR OBJECTIVE FUNCTION
def fitness_function(population, requirements, proficiency_levels):
    
    fitness_values = []
    #any student that is assigned to projects about desired constraints 
    for chromosome in population:
        chromosome_skills = chromosome[:, np.newaxis, :]
        #get the proficiency level for cumulative sum and keep as a variable for further functions
        proficiency = np.sum(chromosome_skills * proficiency_levels * requirements, axis=(1, 2))
        total_proficiency = np.sum(proficiency)

        fitness_values.append(total_proficiency)

    for i, fitness_value in enumerate(fitness_values):
        print(f"Fitness value for Chromosome {i}: {fitness_value}")

    return fitness_values
fitness_values = fitness_function(population,number_of_students,number_of_projects) 

#ELITISM FUNCTION
def elitism(population, fitness_values, elitism_rate):
    
    #Firstly, get the number of elit chromosome from initialized population
    num_elite = int(elitism_rate * len(population))
    #After sort them for the keep best one
    sorted_indices = np.argsort(fitness_values)[::-1]
    elite_population = [population[i] for i in sorted_indices[:num_elite]]
    elite_fitness_values = [fitness_values[i] for i in sorted_indices[:num_elite]]
    for i, chromosome in enumerate(elite_population):
        print(f"Elite Chromosome {i}:")
        print(chromosome)
        print(f"Elite Fitness Value {i}: {elite_fitness_values[i]}")
        print()

    return elite_population, elite_fitness_values

elite_population, elite_fitness_values = elitism(population, fitness_values, elitism_rate)


def selection(population, fitness_values, selection_rate):
    num_parents = int(selection_rate * len(population))
    fitness_sum = np.sum(fitness_values)
    probabilities = fitness_values / fitness_sum

    selected_indices = np.random.choice(len(population), size=num_parents, replace=False, p=probabilities)

    selected_population = [population[idx] for idx in selected_indices]

    for i, chromosome in enumerate(selected_population):
        print(f"Selected Chromosome {i}:")
        print(chromosome)
        print()

    return selected_population
selection_values = selection(population,fitness_values,selection_rate)


def roulette_wheel_selection(population, fitness_values):
    num_parents = len(population)
    fitness_sum = np.sum(fitness_values) 
    probabilities = fitness_values / fitness_sum

    selected_population = []

    for _ in range(num_parents):
        rand_value = random.random() #generating random value between 0 and sum of fitness function
        cumulative_prob = 0.0 #no adding element so cumulative sum is zero

        for i, chromosome in enumerate(population):
            cumulative_prob += probabilities[i]
            if cumulative_prob >= rand_value: #if the cumulative value greter than the random generated one
                selected_population.append(chromosome) #selected one 
                break

    
    for i, chromosome in enumerate(selected_population):
        print(f"Selected Chromosome {i}:")
        print(chromosome)
        print()

    return selected_population

selected_population = roulette_wheel_selection(population, fitness_values)


# use especially when the values of fitnesses are very close for each other
def rank_selection(population, fitness_values):
    num_parents = len(population)
    fitness_sum = np.sum(fitness_values)
    probabilities = np.arange(1, num_parents + 1) / np.sum(np.arange(1, num_parents + 1))

    selected_population = []

    for _ in range(num_parents):
        rand_value = random.random()
        cumulative_prob = 0.0

        for i, chromosome in enumerate(population):
            cumulative_prob += probabilities[i]
            if cumulative_prob >= rand_value:
                selected_population.append(chromosome)
                break

    while len(selected_population) < len(population):
        selected_population.append(random.choice(selected_population))
    
    for i, chromosome in enumerate(selected_population):
        print(f"Selected Chromosome {i}:")
        print(chromosome)
        print()

    return selected_population
selected_population = rank_selection(population,fitness_values)


def tournament_selection(population, fitness_values, tournament_size_percentage):
    num_parents = len(population)
    tournament_size = int(tournament_size_percentage * num_parents)
    selected_population = []

    for _ in range(num_parents):
       
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_population = [population[idx] for idx in tournament_indices]
        tournament_fitness = [fitness_values[idx] for idx in tournament_indices]

       
        winner_idx = np.argmax(tournament_fitness)
        selected_population.append(tournament_population[winner_idx])

    
    for i, chromosome in enumerate(selected_population):
        print(f"Selected Chromosome {i}:")
        print(chromosome)
        print()

    return selected_population
selected_population = tournament_selection(population, fitness_values, tournament_size_percentage)



def crossover(parents):
    offspring = []

    for i in range(0, len(parents) -1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        crossover_point = np.random.randint(1, len(parent1))

        
        child1 = np.concatenate((parent1[:, :crossover_point], parent2[:, crossover_point:]), axis=1)
        child2 = np.concatenate((parent2[:, :crossover_point], parent1[:, crossover_point:]), axis=1)

        offspring.extend([child1, child2])

    
    for i, child in enumerate(offspring):
        print(f"Child {i + 1}:")
        print(child)
        print()

    return offspring
selected_population = tournament_selection(population, fitness_values, tournament_size_percentage)
offspring = crossover(selected_population)

def mutation(population, mutation_rate):
    mutated_population = []

    for chromosome in population:
        mutated_chromosome = chromosome.copy()

        for i in range(mutated_chromosome.shape[0]):
            for j in range(mutated_chromosome.shape[1]):
                if random.random() < mutation_rate:
                    mutated_chromosome[i, j] = 1 - mutated_chromosome[i, j]
                    print(f"Mutation occurred at position ({i}, {j}) in chromosome:")
                    print(mutated_chromosome)
                    print()

        mutated_population.append(mutated_chromosome)

    return mutated_population
mutated_population = mutation(selected_population, mutation_rate)


#number of iterations same as generation number
max_generations = 100  
best_fitness = None
best_chromosome = None

for generation in range(max_generations):
    #Iterate over max generation and apply all the functions
    #Check for the termination condition
    if generation == max_generations - 1:
        
        print("Final Generation:", generation)
        print("Fitness Values:")
        for i, fitness_value in enumerate(fitness_values):
            print(f"Chromosome {i}: {fitness_value}")
            if best_fitness is None or fitness_value > best_fitness:
                best_fitness = fitness_value
                best_chromosome = population[i]
        print("Chromosomes:")
        for i, chromosome in enumerate(population):
            print(f"Chromosome {i}:")
            print(chromosome)
        break


print("Best Fitness:", best_fitness)
print("Best Chromosome:")
print(best_chromosome)
