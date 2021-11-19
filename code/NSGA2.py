# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:50:39 2020

@author: Tiago Santos Ferreira
"""

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_performance_indicator
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


metodo = 1 #mude o numero aqui entre 1 e 3

#-----------------------------------------------------------------------------
if metodo == 1:
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
    print("hv", hv.do(res.F))
    print("hv pareto front:", hv.do(A))

#------------------------------------------------------------------------------

if metodo ==2:
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
    print("hv", hv.do(res.F))
    print("hv pareto front:", hv.do(A))
    
    
#------------------------------------------------------------------------------

if metodo ==3:
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
    print("hv", hv.do(res.F))
    print("hv pareto front:", hv.do(A))
