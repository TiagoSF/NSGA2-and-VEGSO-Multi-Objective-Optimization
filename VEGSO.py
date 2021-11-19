# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:49:49 2020

@author: Tiago Santos Ferreira
"""


import random
import math
import matplotlib.pyplot as plt
import numpy as np

from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_performance_indicator
plt.style.use('seaborn-whitegrid')


metodo = 1 #mude o numero aqui entre 1 e 3

#-----------------------------------------------------------------------------
##ZDT1
if metodo == 1:
    def ZDT_f1(x,y):
        
        z = x
        return z
    
    def ZDT_f2(x,y):
        
        f1 = x
        g = 1 + 9 * (y/(n-1))
        h = 1 - np.sqrt(f1/g)
        f2 = g*h
        
        return f2
    
    # #parametros
    particle_size = 100
    iterations = 200
    rho = 0.3
    gamma = 0.6         #Luciferin enhancement constant
    s = 0.03            #step size
    beta = 0.06         #decision range gain
    n_t = 2             #Desired no. of neighbors
    l_0 = 10            #valor inicial de luciferin
    r_s = 3             #Sensor range
    r_0 = 0.2           #Luciferin decay constant
    n = 30
    
    lower_boundX = 0
    upper_boundX = 1
    lower_boundY = 0
    upper_boundY = 1

#-----------------------------------------------------------------------------
#ZDT2
if metodo == 2:
    def ZDT_f1(x,y):
        
        z = x
        return z
    
    def ZDT_f2(x,y):
        
        f1 = x
        g = 1 + 9 * (y/(n-1))
        h = 1 - ((f1/g)**2)
        f2 = g*h
        
        return f2
    
    # #parametros
    particle_size = 100
    iterations = 200
    rho = 0.3
    gamma = 0.6         #Luciferin enhancement constant
    s = 0.03            #step size
    beta = 0.06         #decision range gain
    n_t = 2            #Desired no. of neighbors
    l_0 = 10          #valor inicial de luciferin
    r_s = 3            #Sensor range
    r_0 = 0.2           #Luciferin decay constant
    n = 30
    
    lower_boundX = 0
    upper_boundX = 1
    lower_boundY = 0
    upper_boundY = 1

#-----------------------------------------------------------------------------
if metodo == 3:
    #ZDT3
    def ZDT_f1(x,y):
        
        z = x
        return z
    
    def ZDT_f2(x,y):
        
        f1 = x
        g = 1 + 9 * (y/(n-1))
        h = 1 - math.sqrt(f1/g) - (f1/g)*math.sin(10*math.pi*f1)
        f2 = g*h
        
        return f2
    
    # #parametros
    particle_size = 100
    iterations = 200
    rho = 0.3
    gamma = 0.6         #Luciferin enhancement constant
    s = 0.03            #step size
    beta = 0.06         #decision range gain
    n_t = 2            #Desired no. of neighbors
    l_0 = 10          #valor inicial de luciferin
    r_s = 3            #Sensor range
    r_0 = 0.2           #Luciferin decay constant
    n = 30
    
    lower_boundX = 0
    upper_boundX = 1
    lower_boundY = 0
    upper_boundY = 1
    


class Glowworms(object):
    def __init__(self, l_0,posX,posY,r_0):
        self.luciferin = l_0
        self.posX = posX
        self.posY = posY
        self.r_0 = r_0

glowworms = []

glowworms1 =[]
glowworms2 = []

# def initialize():
#     for i in range(0,metade):
#         posX = random.uniform(lower_boundX,upper_boundX) #bounds
#         posY = random.uniform(lower_boundY,upper_boundY) #bounds
        
#         glowworms1.append(Glowworms(l_0, posX, posY, r_0))
#     for i in range(metade+1,particle_size):
#         posX = random.uniform(lower_boundX,upper_boundX) #bounds
#         posY = random.uniform(lower_boundY,upper_boundY) #bounds
        
#         glowworms2.append(Glowworms(l_0, posX, posY, r_0))
        
#     glowworms.append(glowworms1)
#     glowworms.append(glowworms2)
    
def initialize():
    for i in range(particle_size):
        posX = random.uniform(lower_boundX,upper_boundX) #bounds
        posY = random.uniform(lower_boundY,upper_boundY) #bounds
        
        glowworms.append(Glowworms(l_0, posX, posY, r_0))
        
        

def distancia(i,j):
    return math.sqrt(((j.posX - i.posX)**2) + ((j.posY - i.posY)**2))  
        
def getNeighborhood(i, glowworms):
    neighborhood = []
    for j in glowworms:
        if (distancia(i,j) < i.r_0) and (i.luciferin < j.luciferin):
            neighborhood.append(j)
    return neighborhood

def sumLuciferin(neighborhood):
    soma = 0
    for i in neighborhood:
        soma = soma + i.luciferin
    return soma

def rouletteSelect(weight):
    weight_sum = 0
    for i in range(len(weight)):
        weight_sum += weight[i]
    
    value = random.uniform(0, 1)*weight_sum
    
    for i in range(len(weight)):
        value -= weight[i]
        if value<=0:
            return i
        
    return len(weight)-1
        

def selectGlowworm(neighborhood, probabilidades):
    index = rouletteSelect(probabilidades)
    
    if len(neighborhood) > 0:
        return neighborhood[index] 
    return 0

def euclideanNorm(x, y):
    dist =  abs(math.sqrt((x**2)+(y**2)))
    if dist == 0:
        return 1
    return dist
        
        
def step(glowworms):
    
    for t in range(iterations):
        # #update luciferin
        # for i in glowworms1:
        #     #minimização
        #     i.luciferin = (1-rho)*i.luciferin - gamma*ZDT1_f1(i.posX, i.posY)
        #     #maximização
        #     #i.luciferin = (1-rho)*i.luciferin + gamma*objective_function(i.posX, i.posY)
        # for i in glowworms2:
        #     #minimização
        #     i.luciferin = (1-rho)*i.luciferin - gamma*ZDT1_f2(i.posX, i.posY)
        #     #maximização
        #     #i.luciferin = (1-rho)*i.luciferin + gamma*objective_function(i.posX, i.posY)
            
        # glowworms.append(glowworms1)
        # glowworms.append(glowworms2)   
        
        
         #update luciferin
        for i in glowworms[0:particle_size//2]:
            #minimização
            i.luciferin = (1-rho)*i.luciferin - gamma*ZDT_f1(i.posX, i.posY)
            #maximização
            #i.luciferin = (1-rho)*i.luciferin + gamma*objective_function(i.posX, i.posY)
        for i in glowworms[((particle_size//2)+1):particle_size]:
            #minimização
            i.luciferin = (1-rho)*i.luciferin - gamma*ZDT_f2(i.posX, i.posY)
            #maximização
            #i.luciferin = (1-rho)*i.luciferin + gamma*objective_function(i.posX, i.posY)
            
        
        #MOVEMENT
        for i in glowworms:
            neighborhood = getNeighborhood(i, glowworms)
            liciferinSumOfNeighborhood = sumLuciferin(neighborhood)
            probabilidades = []
            
            for c in range(len(neighborhood)):
                j = neighborhood[c]
                probabilidade = (j.luciferin - i.luciferin)/(liciferinSumOfNeighborhood - i.luciferin)
                probabilidades.append(probabilidade)
            j = selectGlowworm(neighborhood,probabilidades)
            
            if j != 0:
                oldIPosX = i.posX
                oldIPosY = i.posY
                
                oldJPosX = j.posX
                oldJPosY = j.posY
                
                i.posX = oldIPosX + s*((oldJPosX - oldIPosX) / (euclideanNorm(oldJPosX - oldIPosX, oldJPosY - oldIPosY)));
                i.posY = oldIPosY + s*((oldJPosY - oldIPosY) / (euclideanNorm(oldJPosX - oldIPosX, oldJPosY - oldIPosY)));
                
                if i.posX> upper_boundX:
                    i.posX = upper_boundX
                if i.posX< lower_boundX:
                    i.posX = lower_boundX
                if i.posY> upper_boundY:
                    i.posY = upper_boundY
                if i.posY < lower_boundY:
                    i.posY = lower_boundY
                
                
                # for i in glowworms:
                #     plt.scatter(i.posX, i.posY)
                # plt.show()
                
                
            i.r_0 = min(r_s, max(0, i.r_0 + beta*(n_t - len(neighborhood))) )

            
            
            
            
            
        
initialize()

step(glowworms)


pontos = []

for i in glowworms:
    #print("fitness:", ZDT_f1(i.posX, i.posY), ZDT_f2(i.posX, i.posY))
    pontos.append([ZDT_f1(i.posX, i.posY), ZDT_f2(i.posX, i.posY)])

a = np.array(pontos)
    
ref_point=np.array([1.2, 1.2])


#--------------------------------------------------------------------------------------------------
if metodo == 1:
    problem = get_problem("zdt1")
    pf = problem.pareto_front();
    A = pf[::10] * 1.1
    
    
    plot = Scatter(legend=True,title="ZDT1")
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7,label="Pareto-front")
    plot.add(a,color="red", label="Resultado")
    plot.add(ref_point, color="blue", label="ponto de referência")
    plot.show()
    
    hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
    print("hv", hv.do(a))
    print("hv", hv.do(A))
    
    erro = abs(hv.do(A)-hv.do(a))
    print('erro:', erro)

#-------------------------------------------------------------------------------------------------------
if metodo == 2:
    problem = get_problem("zdt2")
    pf = problem.pareto_front();
    A = pf[::10] * 1.1
    
    plot = Scatter(legend=True,title="ZDT2")
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7,label="Pareto-front")
    plot.add(a,color="red", label="Resultado")
    plot.add(ref_point, color="blue", label="ponto de referência")
    plot.show()
    
    hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
    print("hv", hv.do(a))
    print("hv", hv.do(A))
    
    erro = abs(hv.do(A)-hv.do(a))
    print('erro:', erro)


#----------------------------------------------------------------------------------------------------
if metodo == 3:
    problem = get_problem("zdt3")
    pf = problem.pareto_front();
    A = pf[::10] * 1.1
    
    plot = Scatter(legend=True,title="ZDT3")
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7,label="Pareto-front")
    plot.add(a,color="red", label="Resultado")
    plot.add(ref_point, color="blue", label="ponto de referência")
    plot.show()
    hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
    print("hv", hv.do(a))
    print("hv", hv.do(A))
    
    erro = abs(hv.do(A)-hv.do(a))
    print('erro:', erro)