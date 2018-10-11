#coding:utf-8
from pulp import *

prob = LpProblem('An Integer Programming Demo',LpMaximize)
#step_1:生成变量，但不加入LpProblem中
x1 = LpVariable(name='x1',lowBound=0,cat=LpInteger)
x2 = LpVariable(name='x2',lowBound=0,cat=LpInteger)
x3 = LpVariable(name='x3',lowBound=0,cat=LpInteger)
X = [x1,x2,x3]
prob += x1 + x2 + x3

#step_2:增加目标值
objective = 3*x1 + 4*x2 + 5*x3
#必须用+=的形式，prob = prob + objective会报错
prob += objective

#step_3:增加约束
prob += x1 + 2*x2 <=100
prob += x2 + 3*x3 <=40

status = prob.solve()

print ('最优值为='+str(value(prob.objective)))
#输出所有变量的取值
for i in X:
	print(str(i)+'='+str(value(i)))


