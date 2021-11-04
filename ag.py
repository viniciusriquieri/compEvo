import csv
import time
from numpy.random import rand
from numpy.random import randint
import random
from decimal import *
from collections import namedtuple


file1 = open('films.csv')

listmovies = csv.reader(file1)
header = next(listmovies)
movListId = []
Movies = namedtuple('Movies', ['id', 'name', 'rating', 'duration', 'genre'])

moviesTuple = []

def average(lst):
    return float("{:.2f}".format(sum(lst) / len(lst)))

#define o total de minutos disponiveis diariamente para assistir os filmes
n_watchtime = 240
# define o total de geração
n_gen = 1000
# numero de filmes
n_movies = 93
# tamanho da população (numero de individuos)
n_pop = 300
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_movies)


for movie in listmovies:
    movListId.append(int(movie[0]))
    moviesTuple.append(
        Movies(int(movie[0]), movie[1], float(movie[2]), int(movie[3]),movie[4]))


def formatBestSequence(cromo):
    i = 0
    days = 1
    movTime = 0
    with open('schedule.txt', 'w') as f:
        f.write("----Schedule---- (movieId - movieTitle - duration - rating - genre)\n")
        f.write("Day %d: \n" % days)
        while i < len(cromo):
            movTime += moviesTuple[cromo[i]].duration
            if movTime > n_watchtime:
                f.write("WATCHTIME: %d min \n" % (movTime - moviesTuple[cromo[i]].duration))
                days += 1
                movTime = 0
                f.write("\nDay %d: \n" % days)
            else:
                f.write("%d - %s - %d min - %s - %s\n" % (moviesTuple[cromo[i]].id, moviesTuple[cromo[i]].name, moviesTuple[cromo[i]].duration, moviesTuple[cromo[i]].rating, moviesTuple[cromo[i]].genre))
                i += 1



def fitness(cromo):
    i = 0
    days = 1
    movTime = 0
    if cromo.index(44) > cromo.index(46):
        return 999
    while i < len(cromo):
        movTime += moviesTuple[cromo[i]].duration
        if movTime > n_watchtime:
            days += 1
            movTime = 0
        else:
            i += 1
    return days
#

def generate_pop(pop_size):
    popu = []
    indv = []
    i = 0
    while i < pop_size:
        indv = random.sample(movListId, len(movListId))
        # Se godfather 2 vier antes do 1, um troca a posiçao com o outro
        if indv.index(44) > indv.index(46):
            indv[indv.index(44)] = 46
            indv[indv.index(46)] = 44
            if indv not in popu:  # Verifica se o ultimo individuo gerado já nao consta na população
                popu.append(indv)
                i += 1
    if len(popu) == 1:
        return popu[0]
    return popu
#


def tiebreaker_rating(indv1,indv2):
    i=0
    r1avg = []
    r2avg = []
    tempMean = []
    movTime = 0
    while i < len(indv1):
        movTime += moviesTuple[indv1[i]].duration
        tempMean.append(moviesTuple[indv1[i]].rating)
        if movTime > n_watchtime:
            movTime = 0
            r1avg.append(average(tempMean))
            tempMean = []
        else:
            i += 1    
    tempMean = []
    i=0
    movTime = 0
    while i < len(indv2):
        movTime += moviesTuple[indv2[i]].duration
        tempMean.append(moviesTuple[indv2[i]].rating)
        if movTime > n_watchtime:
            movTime = 0
            r2avg.append(average(tempMean))
            tempMean = []
        else:
            i += 1

    best1 = 0
    best2 = 0

    for i in range(len(r1avg)):
        if r1avg[i]>r2avg[i]: 
            best1+=1
        else: 
            best2+=1

    if best1>best2:
        return indv1
    elif best2>best1:
        return indv2
    else:
        return tiebreaker_genre(indv1,indv2)
#


def tiebreaker_genre(indv1,indv2):
    gscore1 = 1
    gscore2 = 1
    
    for i in range(n_movies):
        if moviesTuple[indv1[i]].genre != moviesTuple[indv1[i-1]].genre:
            gscore1+=1
        if moviesTuple[indv2[i]].genre != moviesTuple[indv2[i-1]].genre:
            gscore2+=1
        
    if gscore1 > gscore2:
        return indv1
    elif gscore2 >= gscore1:
        return indv2
#


## selection operator
def selection_tournament(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]: 
            selection_ix = ix
        # check if scores are equal and perform a tiebreaker
        elif scores[ix] == scores[selection_ix] and scores[ix]!= 999:
            tiebreaker_rating(pop[selection_ix],pop[ix])
    return pop[selection_ix]
#


## crossover 
def crossover(ind1, ind2):
    c1, c2 = [-1] * n_movies, [-1] * n_movies

    xp1,xp2 = random.randint(0, n_movies), random.randint(0, n_movies-1)
    
    if xp2>=xp1:
        xp2+=1
    else:
        xp1,xp2 = xp2, xp1

    if rand() < r_cross:

        #atribui ao filho segmentaçao do meio (entre os pontos de crossover)
        c1[xp1:xp2] = ind1[xp1:xp2]
        c2[xp1:xp2] = ind2[xp1:xp2]

        #duplica os pais em arrays temporarios para manipulaçao sem perder os originais
        ind1t = list(ind1)
        ind2t = list(ind2)

        #preenche segmentaçao final do primeiro filho
        i = xp2
        while i < n_movies:
            while len(ind2t) != 0:
                if ind2t[0] in c1:
                    ind2t.pop(0)
                else:
                    c1[i] = ind2t.pop(0)
                    i+=1
                    break   
        #preenche segmentaçao inicial do primeiro filho
        i = 0
        while i < xp1:
            while len(ind2t) != 0:
                if ind2t[0] in c1:
                    ind2t.pop(0)
                else:
                    c1[i] = ind2t.pop(0)
                    i+=1
                    break
        #preenche segmentaçao final do segundo filho
        i = xp2
        while i < n_movies:
            while len(ind1t) != 0:
                if ind1t[0] in c2:
                    ind1t.pop(0)
                else:
                    c2[i] = ind1t.pop(0)
                    i+=1
                    break
        
        #preenche segmentaçao incial do segundo filho
        i = 0
        while i < xp1:
            while len(ind1t) != 0:
                if ind1t[0] in c2:
                    ind1t.pop(0)
                else:
                    c2[i] = ind1t.pop(0)
                    i+=1
                    break
    else:
        c1,c2 = ind1, ind2

    return [c1, c2]
# 	


## mutation operator
def mutation(cromo, r_mut):
	# check for a mutation
    if rand() < r_mut:
        i = randint(0, n_movies)
        j = randint(0,n_movies)
        if i == j:
            j = randint(0,n_movies)
        
        cromo[i],cromo[j] = cromo[j], cromo[i]
#


# # genetic algorithm
def genetic_algorithm(objective, n_gen, n_pop, r_mut):
    # populaçao inicial de ordem de filmes aleatorios
    pop = generate_pop(n_pop)
    # keep track of best solution
    best, best_eval = pop[0], objective(pop[0])
    # enumerate generations
    for gen in range(n_gen):
        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">New best!! fitness(%s) = %d \nFrom generation %d.\n" % ( pop[i], scores[i], gen))
            if scores[i] == best_eval:
                best_aux = tiebreaker_rating(pop[i], best)
                if best_aux != best:
                    best = best_aux
                    print(">New best by tiebreaker!! fitness(%s) = %d \nFrom generation %d.\n" % ( best, scores[i], gen))
                else:
                    print(">Still the best by tiebreaker!!")
        # select parents
        selected = [selection_tournament(pop, scores) for _ in range(n_pop)]

        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]



start_time = time.time()
# perform the genetic algorithm search
best, score = genetic_algorithm(fitness, n_gen, n_pop, r_mut)
print('############################')
print('Done!')
print()
print('Best offspring: \n%s \nIt takes %d days' % (best, score))
print()
print('OPEN "schedule.txt" FILE TO SEE THE SCHEDULE')
formatBestSequence(best)
print()
print("--- %s seconds ---" % (time.time() - start_time))


