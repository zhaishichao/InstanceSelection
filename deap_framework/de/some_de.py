import random
from array import array
from itertools import chain

import numpy as np
from deap import base
from deap import creator
from deap import tools

def Sum_Of_Squares(x): # x的维度为10，也即D=10
    return [sum(xi ** 2 for xi in x )]





def mutDE(y, a, b, c, f):
    size = len(y)
    for i in range(len(y)):
        y[i] = a[i] + f * (b[i] - c[i])
    return y


def cxBinomial(x, y, cr):
    size = len(x)
    index = random.randrange(size)
    for i in range(size):
        if i == index or random.random() < cr:
            x[i] = y[i]
    return x


IND_SIZE = 5

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array, typecode='d',fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -3, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("mutate", mutDE, f=0.8)

toolbox.register("mate", cxBinomial, cr=0.8)

toolbox.register("select", tools.selRandom, k=3)

toolbox.register("evaluate", Sum_Of_Squares)

def main():
    # Differential evolution parameters
    MU = 5 * 10
    NGEN = 8000

    pop = toolbox.population(n=MU);
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    for g in range(1, NGEN):
        children = []
        for agent in pop:
            # We must clone everything to ensure independence
            a, b, c = [toolbox.clone(ind) for ind in toolbox.select(pop)]
            x = toolbox.clone(agent)
            y = toolbox.clone(agent)
            y = toolbox.mutate(y, a, b, c)
            z = toolbox.mate(x, y)
            del z.fitness.values
            children.append(z)

        fitnesses = toolbox.map(toolbox.evaluate, children)
        for (i, ind), fit in zip(enumerate(children), fitnesses):
            ind.fitness.values = fit
            if ind.fitness > pop[i].fitness:
                pop[i] = ind

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        #print(logbook.stream)

    print("Best individual is ", hof[0])
    print("with fitness", hof[0].fitness.values[0])
    return logbook

if __name__ == "__main__":


    main()