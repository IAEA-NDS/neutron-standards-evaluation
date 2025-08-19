import random
import numpy as np


class Evolution:

    def __init__(self, gene_pool, fitness_func, extra_args=None):
        self._cache = {}
        self._cache['fitness'] = {}
        self._fitness_func = fitness_func
        if extra_args is None:
            extra_args = {}
        self._extra_args = extra_args
        self._gene_pool = gene_pool

    def _normalize_chromosome(self, chromosome):
        return tuple(sorted(set(chromosome)))

    def _prune_chromosome(self, chromosome, keep_genes=None):
        if keep_genes is None:
            keep_genes = []
        chromosome = self._normalize_chromosome(chromosome)
        start_fitness = self.fitness(chromosome)
        pruned_chromosome = list(chromosome)
        i = 0
        while i+1 < len(pruned_chromosome):
            removed_el = pruned_chromosome.pop(i)
            new_fitness = self.fitness(pruned_chromosome) 
            if new_fitness < start_fitness or removed_el in keep_genes:
                pruned_chromosome.insert(i, removed_el)
                i += 1
        return tuple(pruned_chromosome)

    def fitness(self, chromosome):
        chromosome = self._normalize_chromosome(chromosome) 
        if chromosome not in self._cache['fitness']:
            self._cache['fitness'][chromosome] = (
                self._fitness_func(chromosome, **self._extra_args)
            )
        return self._cache['fitness'][chromosome]

    def mutate(self, chromosome):
        start_fitness = self.fitness(chromosome)
        num_new_genes = random.randint(1, 30) 
        gene_pool = list(set(self._gene_pool) - set(chromosome))
        new_genes = random.sample(gene_pool, num_new_genes)
        mc = tuple(chromosome) + tuple(new_genes)
        mc = self._normalize_chromosome(mc)
        mutated_chromosome = self._prune_chromosome(mc, keep_genes=chromosome)
        final_fitness = self.fitness(mutated_chromosome)
        if final_fitness < start_fitness:
            return chromosome
        return mutated_chromosome 

    def crossover(self, chromosomes):
        nc = tuple(el for c in chromosomes for el in c)  
        nc = self._normalize_chromosome(nc)
        new_chromosome = self._prune_chromosome(nc)
        return new_chromosome

    def init_population(self, num_chromosomes):
        population = []
        for i in range(num_chromosomes):
            population.append(self.mutate([]))
        self._population = population

    def next_generation(self):
        population = self._population
        # evaluate fitness of chromosomes in current population
        fitness_scores = [self.fitness(c) for c in population]
        sort_idcs = list(np.argsort(fitness_scores)[::-1])
        inv_sort_idcs = [sort_idcs.index(i) for i in range(len(sort_idcs))]
        # generate the new population
        new_population = []
        # keep the better half of the population (+ mutations)
        for i in range(len(population) // 3):
            curchrom = population[sort_idcs[i]]
            curchrom = self.mutate(curchrom)
            new_population.append(curchrom)
        # generate other half by crossover considering all chromosomes
        popset = set(population)
        sw = [1/(i+1) for i in range(len(population))]
        weights = [sw[inv_sort_idcs[i]] for i in range(len(sw))]
        while len(new_population) < len(population) * 2 // 3:
            chrom1 = random.choices(population, weights=weights, k=1)[0]
            chrom1 = self._normalize_chromosome(chrom1)
            popset.remove(chrom1)
            chrom2 = random.choices(population, weights=weights, k=1)[0]
            chrom2 = self._normalize_chromosome(chrom2)
            popset.add(chrom1)
            new_chromosome = self.crossover([chrom1, chrom2])
            # to avoid inbreeding
            if new_chromosome in popset:
                new_chromosome = self.mutate([])
            new_population.append(new_chromosome)
        # add some new chromosomes
        while len(new_population) < len(population):
            new_population.append(self.mutate([]))
        self._population = new_population

    def average_generation_fitness(self):
        pop = self._population
        return sum(self.fitness(c) for c in pop) / len(pop)

    def get_fittest_chromosomes(self, n=10):
        fitness_scores = [self.fitness(c) for c in self._population]
        order = np.argsort(fitness_scores)[-1:-n:-1]
        return [self._population[i] for i in order]
