# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:58:26 2020

@author: Tiago Santos Ferreira
"""


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_performance_indicator
import numpy as np
import matplotlib.pyplot as plt



def run(metodo):
    
    if metodo == 1:
        plt.style.use('seaborn-whitegrid')
        
        
        problem = get_problem("zdt1")
        
        algorithm = NSGA2(pop_size=100)
        
        res = minimize(problem,
                        algorithm,
                        ('n_gen', 200),
                        verbose=False)
        
        ref_point=np.array([1.2, 1.2])
        
        plot = Scatter(legend=True,title="ZDT1")
        plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7,label="Pareto-front")
        plot.add(res.F, color="red", label="Resultado")
        plot.add(ref_point, color="blue", label="ponto de referência")
        plot.show()
        
        pf = problem.pareto_front();
        
        A = pf[::10] * 1.1
        
        
        hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
        #print("hv", hv.calc(res.F))
        #print("hv", hv.calc(A))
        
        erro = abs(hv.calc(A)-hv.calc(res.F))
        return erro
#------------------------------------------------------------------------------
    if metodo == 2:
        problem = get_problem("zdt2")
        
        algorithm = NSGA2(pop_size=100)
        
        res = minimize(problem,
                        algorithm,
                        ('n_gen', 200),
                        verbose=False)
        
        ref_point=np.array([1.2, 1.2])
        
        plot = Scatter(legend=True, title="ZDT2")
        plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7,label="Pareto-front")
        plot.add(res.F, color="red", label="Resultado")
        plot.add(ref_point, color="blue", label="ponto de referência")
        plot.show()
        
        pf = problem.pareto_front();
        
        A = pf[::10] * 1.1
        
        
        hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
        # print("hv", hv.calc(res.F))
        # print("hv", hv.calc(A))
    
        erro = abs(hv.calc(A)-hv.calc(res.F))
        return erro
#------------------------------------------------------------------------------
    
    if metodo == 3:
        problem = get_problem("zdt3")
        
        algorithm = NSGA2(pop_size=100)
        
        res = minimize(problem,
                        algorithm,
                        ('n_gen', 200),
                        verbose=False)
        
        ref_point=np.array([1.2, 1.2])
        
        plot = Scatter(legend=True, title="ZDT3")
        plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7,label="Pareto-front")
        plot.add(res.F, color="red", label="Resultado")
        plot.add(ref_point, color="blue", label="ponto de referência")
        plot.show()
        
        pf = problem.pareto_front();
        
        A = pf[::10] * 1.1
        
        
        hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
        #print("hv", hv.calc(res.F))
        #print("hv", hv.calc(A))
        
        
        erro = abs(hv.calc(A)-hv.calc(res.F))
        return erro
