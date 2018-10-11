#coding:utf-8

from pulp import *

#LpMinimize是plup类中的常量，plup.LpMinimize
prob = LpProblem('a_lp_problem',LpMinimize)

x11 = LpVariable("x11", lowBound=0, cat='Continuous')
x12 = LpVariable("x12", lowBound=0, cat='Continuous')
x13 = LpVariable("x13", lowBound=0, cat='Continuous')
x14 = LpVariable("x14", lowBound=0, cat='Continuous')
x21 = LpVariable("x21", lowBound=0, cat='Continuous')
x22 = LpVariable("x22", lowBound=0, cat='Continuous')
x23 = LpVariable("x23", lowBound=0, cat='Continuous')
x24 = LpVariable("x24", lowBound=0, cat='Continuous')
x31 = LpVariable("x31", lowBound=0, cat='Continuous')
x32 = LpVariable("x32", lowBound=0, cat='Continuous')
x33 = LpVariable("x33", lowBound=0, cat='Continuous')

X = [x11, x12, x13, x14, x21, x22, x23, x24, x31, x32, x33]
c = [160, 130, 220, 170, 140, 130, 190, 150, 190, 200, 230]
con = [50, 60, 50, 80, 30, 140, 70, 30,10, 50, 10]

z = 0
#X[i]*c[i]返回了pulp.LpAffineExpression类型数据
for i in range(len(X)):
	z += X[i]*c[i]
prob += z

#每一个等式返回一个LpConstraint
#plup里面只支持>=或者<=或者==，不支持>或<
prob += x11+x12+x13+x14 == con[0]# 约束条件1
prob += x21+x22+x23+x24 == con[1]
prob += x31+x32+x33 == con[2]
prob += x11+x21+x31 <= con[3]
prob += x11+x21+x31 >= con[4]
prob += x12 + x22 + x32 <= con[5]
prob += x12 + x22 + x32 >= con[6]
prob += x13 + x23 + x33 <= con[7]
prob += x13 + x23 + x33 >= con[8]
prob += x14 + x24 <= con[9]
prob += x14 + x24 >= con[10]

status = prob.solve()

#prob.objective返回目标函数表达式
#value(prob.objective)返回计算后的目标值
#value函数返回优化后的结果值


print ('最优值为='+str(value(prob.objective)))
#输出所有变量的取值
for i in X:
	print(str(i)+'='+str(value(i)))






