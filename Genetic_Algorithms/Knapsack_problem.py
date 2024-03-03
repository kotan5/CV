# -*- coding: utf-8 -*-

import numpy as np
import random

#%% funcion evalua mochila y penaliza exceso de peso

def evaluaMochila (solucion,valores,pesos,Cmax):
    fit = sum(np.multiply(solucion,valores))
    if (np.multiply(solucion,pesos).sum() > Cmax):
        fit = fit - 1000
    return(fit)


#%% cruce 1-point

def cruce(individuo1, individuo2):
    
    # cruce 1-point 
    # admite la posibilidad de no cruzar para nada 
    # eso es, point 0 y 4
    
    point = np.random.randint(0, np.size(individuo1))
    
    hijos = np.zeros((2,np.size(individuo1)))
    
    hijos[0,0:point] = individuo1[0:point]
    hijos[1,0:point] = individuo2[0:point]
    
    hijos[0,point:] = individuo2[point:]
    hijos[1,point:] = individuo1[point:]
    
    return hijos

#%% mutacion bit-flip

def mutacion(individuos, pm):
    
    for aa in range(np.size(individuos[0,:])): #columnas
        for bb in range(np.size(individuos[:,1])): #filas
            
            if np.random.random() < pm:
                if individuos[bb,aa] == 0:
                    individuos[bb,aa] = 1
                else:
                    individuos[bb,aa] = 0
                          
    return individuos

#%% supervivientes

def supervivientes(pob1, fit1, pob2, fit2, numSuper):
    
    total = np.concatenate((pob1,pob2))
    fittotal = np.concatenate((fit1,fit2))
    
    aux = (np.argsort(fittotal))[-numSuper:]
    
    superviv = total[aux,:]
    fitSuperviv = fittotal[aux]
    
    return superviv, fitSuperviv

#%% funcion principal

def GA_mochila(popSize,numPadres,numIteraciones,pm,numSuperviv):
    
    # datos iniciales del problema
    vSize = 5
    pesos = np.array([1, 2, 1.5, 3, 5])
    valores = np.array([7, 9, 10, 10, 8])
    Cmax = 7
    
    # Poblacion inicial
    poblacion = np.zeros([popSize,vSize])
    fitness = np.zeros(popSize)
    
    for i in range(popSize):
        poblacion[i,:] = random.choices(range(2),k=vSize)
        fitness[i] = evaluaMochila(poblacion[i,:],valores,pesos,Cmax)

        
    # Bucle principal. Criterio de parada: número de iteraciones

    for jj in range(numIteraciones):
        
        matrizHijos = np.zeros([numPadres,vSize])
        
        for kk in range(int(numPadres/2)):
            
            #Seleccion de padres torneo (cojo 2 y me quedo con el mejor, dos veces)
            quienes = random.sample(range(popSize),2)
            v = np.argsort(fitness[quienes]) # Me ordena de menor a mayor y me devuelve los índices
            u = v[-1] # Me interesa el mayor, o sea, el ultimo
            candidato1 = quienes[u] 
            
            quienes = random.sample(range(popSize),2)
            v = np.argsort(fitness[quienes]) # Me ordena de menor a mayor y me devuelve los índices
            u = v[-1] # Me interesa el mayor, o sea, el último
            candidato2 = quienes[u]

            #Crossover
            hijos = cruce(poblacion[candidato1,],poblacion[candidato2,])
    
            #mutacion
            hijos = mutacion(hijos,pm)

            #Guardo los hijos
            matrizHijos[(kk*2):(kk*2+2),:] = hijos
        
        #Evaluacion de hijos
        fitnessHijos = np.zeros(numPadres)
    
        for i in range(numPadres):
            fitnessHijos[i] = evaluaMochila(matrizHijos[i,], valores, pesos, Cmax)
     
        #Seleccion supervivientes
        [poblacion,fitness] = supervivientes(poblacion,fitness,matrizHijos,fitnessHijos,numSuperviv)
            
            
        #Imprime info
            
        print("Iteración ",jj,": min = ",min(fitness), ", max = ", max(fitness),", media = ",np.mean(fitness))

        
    # Resultado final
        
    i = np.argmax(fitness)
    print(poblacion[i,:])
    print('Peso de mochila: ', sum(np.multiply(poblacion[i,:], pesos)))

#%% datos del problema
vSize = 5
pesos = np.array([1, 2, 1.5, 3, 5])
valores = np.array([7, 9, 10, 10, 8])
Cmax = 7

popSize = 4
numIteraciones = 2
numPadres = 4
pm = 0.5 # probabilidad de mutacion
numSuperviv = 4 # cuantos sobreviven en cada iteracion

#%%

GA_mochila(popSize, numPadres, numIteraciones, pm, numSuperviv)