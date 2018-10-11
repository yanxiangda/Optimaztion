from cvxopt import matrix, solvers
import pandas as pd
import numpy as np
from random import uniform
import random
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import re
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

#将属性数据整理为数据表的格式
def paste_attr(training_data_attr_,training_data,col_name):
	#注意要用两个中括号，如果只用一个中括号的话，就会返回series
	training_data_dim = training_data[['sku_id']].drop_duplicates()
	#很奇怪，会出现一个sku的某一个属性对应多个属性取值的情况，遇到这种属性，直接忽略掉这个属性，而不是去掉sku全部的属性
	#training_data = training_data[['sku_id','com_attr_name','com_attr_value_name']]
	training_data_filter = training_data.groupby(['sku_id','com_attr_name']).count().reset_index()
	training_data_filter.columns = ['sku_id','com_attr_name','num']
	training_data_filter = training_data_filter[training_data_filter.num==1]
	#drop函数不要连接在后面写，容易出问题
	training_data_filter = training_data_filter.drop('num',axis=1)
	training_data = pd.merge(training_data,training_data_filter,on=['sku_id','com_attr_name'],how='right')
	attr_size = training_data_attr_.loc[:,col_name].size
	for i in range(attr_size):
		current_attr_name = training_data_attr_.iloc[i,0]
		training_data_sub = training_data[training_data.com_attr_name==current_attr_name][['sku_id','com_attr_value_name']]
		#pandas.DataFrame.join操作的时候有问题，而且必须制定suffix
		training_data_dim = pd.merge(training_data_dim,training_data_sub,on='sku_id',how='left')
		training_data_dim.rename(columns={'com_attr_value_name':current_attr_name},inplace=True)
	return training_data_dim


#普通线性回归与lasso回归
def linear_reg_cv(df,if_log=True,model='LinearRegression'):
	random_list = [uniform(0, 1) < 0.8 for _ in range(len(df))]
	df = df.drop(['sku_id','sale_qtty'],axis=1)
	df_training = df[random_list]
	df_test = df[[not a for a in random_list]]
	y_training = list(df_training['trans_price'])
	y_test = list(df_test['trans_price'])
	if if_log==True:
		y_training = [math.log(y) for y in y_training]
		y_test = [math.log(y) for y in y_test]
	del df_training['trans_price']
	del df_test['trans_price']
	x_training = np.asmatrix(df_training)
	x_test = np.asmatrix(df_test)
	#reg = linear_model.LinearRegression()
	reg = None
	if model=='LinearRegression':
		reg = linear_model.LinearRegression()
	elif model=='Lasso':
		reg = linear_model.Lasso(alpha=0.001)
	else:
		reg = linear_model.LinearRegression()
	reg.fit(x_training, y_training)
	y_hat_training = reg.predict(x_training)
	y_hat_test = reg.predict(x_test)
	k = 0
	x_list = list(df_training.columns)
	for i in range(len(reg.coef_)):
		if reg.coef_[i] != 0:
			print(i)
			print(x_list[i])
			print(reg.coef_[i])
			k += 1
	print(k)
	mse_coef = list(np.array(x_training.sum(axis=0))[0])
	w = list(reg.coef_)
	'''
	coef_mse = []
	for i in range(len(mse_coef)):
		if w[i] == 0:
			coef_mse.append(0.0)
		else:
			m = mse_coef[i]
			coef_mse.append(1.96*math.sqrt(mean_squared_error(y_training, y_hat_training))/math.sqrt(m*(1-m/10000)**2+(m/10000)**2*(10000-m)))
	coef_mse = np.array(coef_mse)
	'''
	return pd.Series({'coef': reg.coef_, 'coef_mse': None, 'interc': reg.intercept_, 'mse': mean_squared_error(y_training, y_hat_training),
						'r2': r2_score(y_training, y_hat_training), 'mape': mean_absolute_error(y_training, y_hat_training),
						'mse_test': mean_squared_error(y_test, y_hat_test), 'r2_test': r2_score(y_test, y_hat_test),
						'mape_test': mean_absolute_error(y_test, y_hat_test)})

def add_con(feature_list_,revised_feature_list_):
	global_index_ = 0
	split_list_ = list()
	index_ = 0
	for i in feature_list_:
		sp = i.split('__')
		sp.append(index_)
		split_list_.append(sp)
		index_ = index_+1
	split_df_ = pd.DataFrame(split_list_)
	split_df_.columns = ['feature_name','feature_value','feature_index']
	#新建全零的矩阵
	row_size = 0
	for i in revised_feature_list_:
		split_df_current_ = split_df_[split_df_.feature_name==i]
		row_size = row_size + len(split_df_current_) - 1
	g = np.zeros([row_size,len(feature_list_)],dtype='double')
	for i in revised_feature_list_:
		split_df_current_ = split_df_[split_df_.feature_name==i]
		try:
			split_df_current_['feature_value'] = split_df_current_['feature_value'].astype('int64')
		except:
			print ('不能被转换为int')
		split_df_current_ = split_df_current_.sort_values('feature_value').reset_index(drop=False)
		for j in range(len(split_df_current_)-1):
			g[global_index_][split_df_current_.loc[j,'feature_index']] = 1
			g[global_index_][split_df_current_.loc[j+1,'feature_index']] = -1
			global_index_ = global_index_+1
	return g,row_size

#二次规划函数
def linear_reg_cv_qp(df,feature_list,if_log=True,alpha=0.005):
	#df['itce'] = 1
	#print(df)
	random_list = [uniform(0, 1) < 0.8 for _ in range(len(df))]
	df = df.drop(['sku_id','sale_qtty'],axis=1)
	df_training = df[random_list]
	df_test = df[[not a for a in random_list]]
	Y_training = list(df_training['trans_price'])
	Y_test = list(df_test['trans_price'])
	Y_training_ = Y_training
	if if_log==True:
		Y_training_ = [math.log(y) for y in Y_training]
		Y_test = [math.log(y) for y in Y_test]
	Y_training = matrix(Y_training_) 
	del df_training['trans_price']
	del df_test['trans_price']
	X_training_ = np.asmatrix(df_training)
	X_training = matrix(np.asmatrix(df_training).tolist()).T
	X_list = list(df_training.columns)
	X_test = matrix(np.asmatrix(df_test).tolist()).T
	re = matrix(np.eye(len(df_training.columns)))
	p = X_training.T * X_training + alpha*re
	print (len(Y_training))
	print (len(X_training))
	q = (-1.0 * Y_training.T * X_training).T
	#G = [[0 for _ in range(130)] for __ in range(37)]
	#G = matrix(define_g(G)).T
	G,h_size_ = add_con(df_training.columns.tolist(),feature_list)
	G = matrix(G.tolist()).T
	h = [-0.05 for _ in range(h_size_)]
	h = matrix(h)
	sol = solvers.qp(p, q, G, h)
	features_list_ = list()
	for i in range(len(df_training.columns)):
		print(i)
		print(list(sol['x'].T)[i])
		print(X_list[i])
		features_list_.append([i,list(sol['x'].T)[i],X_list[i]])
	w = sol['x']
	f = X_training*w
	f_list = list(f.T)
	f_test = list((X_test*w).T)
	#mse_coef = list(np.array(X_training_.sum(axis=0))[0])
	'''
	coef_mse = []
	for i in range(len(mse_coef)):
		m = mse_coef[i]
		if m == 0:
			coef_mse.append(0.0)
		else:
			coef_mse.append(1.96*math.sqrt(mean_squared_error(Y_training,f_list))/math.sqrt(m*(1-m/300000)**2+(m/300000)**2*(300000-m)))
	coef_mse = np.array(coef_mse)
	'''
	return features_list_,pd.Series({'mse': mean_squared_error(f_list, Y_training), 'r2': r2_score(f_list, Y_training_),
						'mape': mean_absolute_error(f_list, Y_training_), 'mse_test': mean_squared_error(f_test, Y_test),
						'r2_test': r2_score(f_test, Y_test), 'mape_test': mean_absolute_error(f_test, Y_test)})




'''
==================================================================================
==                                                                             	==
==                               Spark获取数据                                	==
==                                                                             	==
==================================================================================
'''
attr_data_ = spark.sql('''select 
			item_sku_id,com_attr_name,coalesce(com_attr_value_name,com_attr_value_rem) as com_attr_value_name 
			from gdm.gdm_m03_item_sku_spec_par_da 
			where dt='2017-05-31' and data_type='1' and cate_id='679'
			''')
#订单数据避开2017-06月份，获取两年的数据
ord_data_ = spark.sql('''
			select distinct sku_id,sale_qtty,trans_price 
			from dev.dp_pl_es_ext 
			where dt<='2018-05-31' and dt>='2015-06-01' and cid3='679'
			''')
'''
==================================================================================
==                                                                             	==
==                             数据预处理与转换                               	==
==                                                                             	==
==================================================================================
'''
#os.chdir('D:/Report/Price_Structure/2018-04-23')
#training_data = pd.read_csv('demo_679_r.csv',header=0,sep=',',encoding='gb18030').drop_duplicates()
training_data = attr_data_.toPandas()
training_data.columns=['sku_id', 'com_attr_name', 'com_attr_value_name']
#training_data = training_data[['sku_id','com_attr_name','com_attr_value_name']].drop_duplicates()
#获取属性集，将弹性化的表转为结构化的表
training_data_attr_ = training_data[['com_attr_name']].drop_duplicates().reset_index(drop=True)
training_data_ = paste_attr(training_data_attr_,training_data,col_name='com_attr_name')

#只分析核心品牌为Nvidia，共有3002行数据
training_data_ = training_data_[training_data_[u'核心品牌']=='NVIDIA']
training_data_ = training_data_.dropna(subset=[u'核心型号'])

training_data_[u'核心型号_re'] = [str(_).lower() for _ in training_data_[u'核心型号']] 
training_data_[u'核心型号_re'] = [_.replace('geforce','').replace('nvidia','').replace(' ','') for _ in training_data_[u'核心型号_re']]

#读取标准的nvidia型号显卡
nvidia_ = pd.read_csv('nvidia_list.csv',encoding='utf-8')
nvidia_list_ = list()
for x in range(len(nvidia_.columns)):
	for y in (nvidia_.iloc[:,x].dropna()):
		nvidia_list_.append([nvidia_.columns[x],y])

nvidia_df_ = pd.DataFrame(nvidia_list_,columns=['gernation','version_complete'])
nvidia_df_['version_complete_lower'] = [_.lower().replace(' ','') for _ in nvidia_df_['version_complete']]


training_data_ = pd.merge(training_data_,nvidia_df_,left_on=u'核心型号_re',right_on='version_complete_lower',how='left')
training_data_ = training_data_.dropna(subset=['version_complete_lower'])

usefull_columns = ['sku_id',u'品牌',u'最大分辨率',u'核心频率',u'散热器类型',u'显存容量','version_complete_lower','gernation']
final_training_data_ = training_data_[usefull_columns].dropna()
final_training_data_ = final_training_data_.rename(columns={u'品牌':u'brand',u'最大分辨率':'max_resolution',u'核心频率':'core_freqency',u'散热器类型':'colder_type',u'显存容量':'memory_size'})
#处理品牌
final_training_data_['brand'] = [re.match(u'[\u4e00-\u9fa5]*',_).group() for _ in final_training_data_['brand']]
#处理核心频率
final_training_data_['core_freqency_max'] = [__ for __ in (np.max(map(int,re.findall('\d+',_))) for _ in final_training_data_['core_freqency'])]
final_training_data_['core_freqency_min'] = [__ for __ in (np.min(map(int,re.findall('\d+',_))) for _ in final_training_data_['core_freqency'])]
#处理显存容量
final_training_data_['memory_size'] = [re.findall('\d*[g,m]',_.replace(' ','').lower())[0] for _ in final_training_data_['memory_size']]
return_list = list()
for i in final_training_data_['memory_size']:
	cr = re.match('\d*g',i)
	if cr==None:
		return_list.append(int(re.match('\d*',i).group()))
	else:
		return_list.append(int(re.match('\d*',i).group())*1024)

final_training_data_['memory_size'] = return_list
#处理最大分辨率
final_training_data_['max_resolution'] = [np.nan if re.match('\d{4}[×,x,*]\d{4}',_.replace(' ',''))==None else re.match('\d{4}[×,x,*]\d{4}',_.replace(' ','')).group().replace('×','x').replace('*','x') for _ in final_training_data_['max_resolution']]

final_used_columns = ['sku_id','brand','core_freqency_max','core_freqency_min','colder_type','memory_size','version_complete_lower']
final_training_data_ = final_training_data_[final_used_columns].replace('',np.nan).dropna()
final_training_data_ = final_training_data_.sort_values(by=['memory_size'])
final_training_data_['memory_size'] = final_training_data_['memory_size'].astype('str')
#one_hot编码后，只有4个属性作为最后分析的特征
final_training_data_one_hot_ = pd.get_dummies(final_training_data_[['brand','memory_size','colder_type','version_complete_lower']],prefix=['brand_','memory_size_','colder_type_','version_complete_lower_'])
final_training_data_one_hot_['sku_id'] = final_training_data_['sku_id']
final_training_data_['sku_id'] = final_training_data_['sku_id'].astype('int')
ord_data_['sku_id'] = ord_data_['sku_id'].astype('int')
ord_data_ = ord_data_.toPandas()
final_training_data_one_hot_ = pd.merge(final_training_data_one_hot_,ord_data_,on='sku_id',how='left').dropna()


#按照sale_qtty为权重进行抽样
dataset_size_ = final_training_data_one_hot_.iloc[:,0].size
#抽样2/3,以sale_qtty为权重
dataset_sample_ = final_training_data_one_hot_.sample(n=int(dataset_size_*5),weights=final_training_data_one_hot_['sale_qtty'],replace=True)
dataset_sample_ = dataset_sample_.reset_index(drop=True)

#re_1 = linear_reg_cv(dataset_sample_,True,'LinearRegression')
features_list_,re_2 = linear_reg_cv_qp(dataset_sample_,['memory_size'],True,alpha=0.05)

