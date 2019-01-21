# Import Statemens

import numpy as np 
import statistics
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as path_effects
from matplotlib import rcParams
import sys
import matplotlib

# Setting some default setting
matplotlib.rcParams.update({'figure.max_open_warning': 0})

sys.setrecursionlimit(1000000)
reload(sys)  
sys.setdefaultencoding('utf8')
# Font properties Change
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams.update({'font.size':10})

# function to print array
def print_arr(arr):
	count = 0
	for i in arr:
		if (len(i) != 11):
			count += 1
		print (i)
	print(count)

# funcion to convert array
def convertMat(mat):
	outMat = []
	rows = len(mat[0])
	cols = len(mat)
	for i in range(rows):
		temp = []
		for j in range(cols):
			temp.append(mat[j][i])
		outMat.append(deepcopy(temp))
	return outMat

# function to calc;uate mean
def mean(l):
	return sum(l)/len(l)


# function to partition dataset
def partition_dataset(dataset,output_dataset,percentage):
	len_dataset = len(dataset)
	len_new_arr = int(percentage*float(len_dataset)/100)
	new_dataset = []
	new_dataset_output = []
	dataset_1 = []
	output_dataset_1 = []
	tmp_arr = []
	for i in range(len_new_arr):
		temp = random.randint(1,len_dataset-1)
		if temp in tmp_arr:
			while temp in tmp_arr:
				temp = random.randint(1,len_dataset-1)
		tmp_arr.append(temp)
	tmp_arr.sort()
	j = 0
	for i in range(len(dataset)):
		if (j>= len(tmp_arr)):
			dataset_1.append(deepcopy(dataset[i]))
			output_dataset_1.append(deepcopy(output_dataset[i]))
		elif (i == tmp_arr[j]):
			new_dataset.append(deepcopy(dataset[i]))
			new_dataset_output.append(deepcopy(output_dataset[i]))
			j += 1
		else :
			dataset_1.append(deepcopy(dataset[i]))
			output_dataset_1.append(deepcopy(output_dataset[i]))
	return dataset_1,output_dataset_1,new_dataset,new_dataset_output

# function to read data and populate the dataset
def Read_data(filepath):
	num_dataset = 0
	data_arr = []
	output_arr = []

	with open(filepath) as fp:
		line = fp.readline()
		while (line):
			temp_arr = line.split(',')
			temp_output = temp_arr[-1]
			del temp_arr[-1:]
			output_arr.append(float(temp_output))
			if (temp_arr[0] == 'M'):
				del temp_arr[0:1]
				temp_arr.append(0)
				temp_arr.append(0)
				temp_arr.append(1)
			elif (temp_arr[0] == 'I'):
				del temp_arr[0:1]
				temp_arr.append(0)
				temp_arr.append(1)
				temp_arr.append(0)
			elif (temp_arr[0] == 'F'):
				del temp_arr[0:1]
				temp_arr.append(1)
				temp_arr.append(0)
				temp_arr.append(0)
			line = fp.readline()
			num_dataset = num_dataset + 1
			temp_arr = list(map(float, temp_arr))
			data_arr.append(temp_arr)
	return data_arr,output_arr

# function to stansadrise the dataset
def standardize_dataset(data_arr):
	mean_arr = []
	sd_arr = []
	for i in range(len(data_arr)):
		tmp_mean = mean(data_arr[i])
		tmp_sd = statistics.stdev(data_arr[i])
		mean_arr.append(tmp_mean)
		sd_arr.append(tmp_sd)
		for j in range(len(data_arr[i])):
			data_arr[i][j] = (data_arr[i][j] -tmp_mean)/tmp_sd
	return data_arr,mean_arr,sd_arr

# function to standadrise test dataset
def standardize_test_dataset(data_arr,mean_arr,sd_arr):

	for i in range(len(data_arr)):
		for j in range(len(data_arr[i])):
			data_arr[i][j] = (data_arr[i][j] - mean_arr[i])/sd_arr[i]
	return data_arr

# function to caclulate norm of w
def norm(w):
	sum = 0
	for i in range(len(w)):
		sum += w[i]*w[i]
	return sum

# function to cacluate error
def calc_error(X,Y,w,tmp_lambda):
	tmp_sum = 0
	f_of_x = X.dot(w)
	
	for i in range(len(X)):	
		tmp_sum = tmp_sum + (f_of_x[i] - Y[i])*(f_of_x[i] - Y[i])

	return tmp_sum/(2*len(X)) + tmp_lambda*norm(w)

# function to caclutate gradent descent
def calc_diff(X,Y,w,index):
	tmp_sum = 0
	f_of_x = X.dot(w)

	for i in range(len(X)):
		tmp_sum = tmp_sum + (f_of_x[i] - Y[i])*X[i][index]
		
	return tmp_sum

# function to perform linear regrssion
def mylinridgereg_1(X,Y,tmp_lambda):
	ideintity_matrix = []
	for i in range(len(X[0])):
		tmp_arr = []
		for j in range(len(X[0])):
			if (i==j):
				tmp_arr.append(tmp_lambda)
			else :
				tmp_arr.append(0)
		ideintity_matrix.append(tmp_arr)


	ideintity_matrix = np.asarray(ideintity_matrix)
	w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X) + ideintity_matrix), np.transpose(X)).dot(Y)
	return w

# function to perform linear regrssion
def mylinridgereg(X,Y,tmp_lambda):
	alpha = 0.00001
	num_itter = 0
	w = []
	if (len(X) != 0):
		for i in range(len(X[0])):
			w.append(0)
	else :
		print("Invalid Dataset... Exiting !!!!")
		exit()

	w = np.asarray(w, dtype = float)
	tmp_error = calc_error(X,Y,w,tmp_lambda)

	for i in range(len(w)):
		w[i] = w[i] - alpha*(calc_diff(X,Y,w,i) + tmp_lambda*2*w[i])
		
	new_error = calc_error(X,Y,w,tmp_lambda)
	
	print(new_error,tmp_error),

	while( num_itter < 1000):
		tmp_error = new_error
		for i in range(len(w)):
			# print ("w is "+ str(w))
			w[i] = w[i] - alpha*(calc_diff(X,Y,w,i) + tmp_lambda*2*w[i])
		new_error = calc_error(X,Y,w,tmp_lambda)
		num_itter += 1

	print(new_error,tmp_error)
	print("num itteration = " + str(num_itter))
	return w


# function to caclulate mean squared error
def meansquarederr(T, Tdash):
	sum_out = 0
	if (len(T) != len(Tdash)):
		print("output len missmatch error Exiting !!!!!")
		exit()
	for i in range(len(T)):
		sum_out += (T[i]-Tdash[i])*(T[i]-Tdash[i])
	return float(sum_out)/len(T)


# makinf number of fraction array
num_fraction = 110
frac_arr = [(float(x)+1)/(num_fraction) for x in range(num_fraction) ]
frac_arr = frac_arr[:-int(0.09*100)]

# frac_arr = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]

num_random_dataset = 1
# 
filepath = "./l2/linregdata"
data_arr,output_arr = Read_data(filepath)
data_arr_backup = deepcopy(data_arr)
output_arr_backup = deepcopy(output_arr)


error_carr_traning = []
error_carr_test = []
number_points_arr_traning = []
number_points_arr_test = []

lambda_min_arr = []
lambda_min_arr_val  = []
lambda_index_arr = []
overall_min = 100000
overall_index = [-1,-1]


# for loop to perform test on fraction arr.
for i in range(len(frac_arr)):
	min_val = 1000000
	temp_lambda = -1
	min_index = -1
	print("------- frac = "+ str(frac_arr[i]))
	n = 200
	lambda_arr = [float(x)+1 for x in range(n)]

	traning_dataset,traning_output,test_dataset,test_output = partition_dataset(data_arr,output_arr,20)
	traning_dataset,traning_output,validation_dataset,validation_output = partition_dataset(traning_dataset,traning_output,frac_arr[i]*100)

	number_points_arr_traning.append(len(traning_dataset))
	number_points_arr_test.append(len(test_dataset))
	
	traning_dataset = convertMat(traning_dataset)
	traning_dataset,mean_arr,sd_arr = standardize_dataset(traning_dataset)
	traning_dataset = convertMat(traning_dataset)

	test_dataset = convertMat(test_dataset)
	test_dataset = standardize_test_dataset(test_dataset,mean_arr,sd_arr)
	test_dataset = convertMat(test_dataset)

	for k in range(len(traning_dataset)):
		traning_dataset[k].insert(0,1)
		
	for k in range(len(test_dataset)):
		test_dataset[k].insert(0,1)

	for k in range(len(validation_dataset)):
		validation_dataset[k].insert(0,1)

	traning_dataset = np.asarray(traning_dataset, dtype = float)
	traning_output = np.asarray(traning_output, dtype = float)
	
	test_dataset = np.asarray(test_dataset, dtype = float)
	test_output = np.asarray(test_output, dtype = float)

	validation_dataset = np.asarray(validation_dataset, dtype = float)
	validation_output = np.asarray(validation_output, dtype = float)

	error_arr_frac_traning = []
	error_arr_frac_test = []
	for j in range(len(lambda_arr)):
		average_err_traning = 0
		average_err_test = 0 


		tmp_arr = mylinridgereg_1(traning_dataset,traning_output,lambda_arr[j])
		
		f_of_x = test_dataset.dot(np.asarray(tmp_arr))
		f_of_x_traning = traning_dataset.dot(np.asarray(tmp_arr))

		error_arr_frac_traning.append(meansquarederr(f_of_x_traning,traning_output))
		error_arr_frac_test.append(meansquarederr(f_of_x,test_output))

		if (error_arr_frac_test[j] < min_val):
			temp_lambda = lambda_arr[j]
			min_val = error_arr_frac_test[j]
			min_index = j

		if (error_arr_frac_test[j] < overall_min):
			overall_min = error_arr_frac_test[j]
			overall_index = [i,j]
			test_output_plot = deepcopy(test_output)
			traning_output_plot = deepcopy(traning_output)
			predecited_output = test_dataset.dot(np.asarray(tmp_arr))
			predecited_output_traning = traning_dataset.dot(np.asarray(tmp_arr))
		

	if (i%10 == 0):

		fig, ax = plt.subplots()
		ax.yaxis.grid(True)
		ax.set_axisbelow(True)

		ax.minorticks_on()
		ax.tick_params(axis='x',which='minor',bottom='off')
		ax.yaxis.grid(True)
		ax.set_axisbelow(True)

		ax.plot(lambda_arr,error_arr_frac_traning,label = r'Traning Error')
		ax.plot(lambda_arr,error_arr_frac_test,label = r'Test Error')

		ax.set_xlabel(r'$\lambda$',fontsize=15)
		ax.set_ylabel("Error",fontsize=15)
		ax.set_title('Dependence of Error with lambda\n(fraction = ' + str(frac_arr[i]) + ')',fontweight= 'bold',fontsize=15)
		ax.legend(loc="best")
		plt.tight_layout()
		plt.savefig("lambda_vs_error_" + str(frac_arr[i]) +".pdf", bbox_inches='tight')
		# plt.show()

	error_carr_traning.append(error_arr_frac_traning)
	error_carr_test.append(error_arr_frac_test)
	lambda_min_arr.append(min_val)
	lambda_min_arr_val.append(temp_lambda)
	print("Minimum error on all values of lambda for fraction =  " + str(frac_arr[i]) +  ", is " + str(min_val))
	# print("Average error on all values of lambda is " + str(min_val))
	print("--------------\n\n")


# ********* Error vs Fraction *******************
fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)

ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)

for i in range(len(frac_arr)):
	ax.plot(lambda_arr,error_carr_traning[i],label = 'frac = ' + str(frac_arr[i]))


ax.set_title('Dependence of Traning Error with different fraction of\nValidation/Traning dataset',fontweight= 'bold',fontsize=15)
ax.set_xlabel(r'$\lambda$',fontsize=15)
ax.set_ylabel("Error",fontsize=15)
plt.tight_layout()
# plt.savefig("frac_dependence"+".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig("frac_dependence"+".pdf", bbox_inches='tight')


# ************* Error vs fraction ***********************

fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)

ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)


for i in range(len(frac_arr)):
	ax.plot(lambda_arr,error_carr_test[i],label = 'frac = ' + str(frac_arr[i]) )


ax.set_title('Dependence of Test Error with different fraction of\nValidation/Traning dataset',fontweight= 'bold',fontsize=15)
ax.set_xlabel(r'$\lambda$',fontsize=15)
ax.set_ylabel("Error",fontsize=15)
# ax.legend(loc="best")
# handles, labels = ax.get_legend_handles_labels()
# lgd = ax.legend(handles, labels, loc='center left',bbox_to_anchor=(1.05, 0.5)) 
plt.tight_layout()
# plt.savefig("frac_dependence_test"+".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig("frac_dependence_test"+".pdf", bbox_inches='tight')

# Lambda vs Fraction

fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)

ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)

ax.plot(frac_arr,lambda_min_arr_val)

ax.set_title(r'Lambda vs Fraction',fontweight= 'bold',fontsize=15)
ax.set_ylabel(r'Lambda',fontsize=15)
ax.set_xlabel("Traning Set Fraction",fontsize=15)
# # ax.legend(loc="best")
# handles, labels = ax.get_legend_handles_labels()
# lgd = ax.legend(handles, labels, loc='center left',bbox_to_anchor=(1.05, 0.5)) 
plt.tight_layout()
plt.savefig("lambda_vs_frac"+".pdf", bbox_inches='tight')

# **************** Mean Error vs fraction ****************

fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)

ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)

ax.plot(frac_arr,lambda_min_arr)

ax.set_title('Minimum average mean squared \ntesting error vs Fraction',fontweight= 'bold',fontsize=15)
ax.set_ylabel('Minimum average mean \nsquared testing error',fontsize=15)
ax.set_xlabel("Traning Set Fraction",fontsize=15)
# # ax.legend(loc="best")
# handles, labels = ax.get_legend_handles_labels()
# lgd = ax.legend(handles, labels, loc='center left',bbox_to_anchor=(1.05, 0.5)) 
plt.tight_layout()
plt.savefig("meanError_vs_frac"+".pdf", bbox_inches='tight')


# ************** Predected vs Original *******************

fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)

ax.minorticks_on()
# ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)

x = np.linspace(0, int(max(test_output_plot)),100) 
ax.plot(test_output_plot,predecited_output,color = 'C1',marker = '.', linestyle = '',markersize = 3)
ax.plot(x,x,color = 'blue' )

ax.set_title('Predicted output vs Actual output for Test Dataset\n(lambda = ' + str(lambda_arr[overall_index[1]]) +  ', fraction = ' + str(frac_arr[overall_index[0]]) +')',fontweight= 'bold',fontsize=15)
ax.set_ylabel(r'Predicted Target Value',fontsize=15)
ax.set_xlabel('Actual Target Value',fontsize=15)
ax.axis('equal')
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('scaled')
# ax.legend(loc="best")
# handles, labels = ax.get_legend_handles_labels()
# lgd = ax.legend(handles, labels, loc='center left',bbox_to_anchor=(1.05, 0.5)) 
plt.tight_layout()
plt.savefig("Predicted_vs_original_test"+".pdf", bbox_inches='tight')

# ******************* Predected vs Traning ***********************

fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)

ax.minorticks_on()
ax.yaxis.grid(True)
ax.set_axisbelow(True)

x = np.linspace(0, int(max(traning_output_plot)),100) 

ax.plot(traning_output_plot,predecited_output_traning,color = 'C1',marker = '.', linestyle = '',markersize = 3)
ax.plot(x,x,color = 'blue' )

ax.set_title('Predicted output vs Actual output for Traning Dataset\n(lambda = ' + str(lambda_arr[overall_index[1]]) +  ', fraction = '+ str(frac_arr[overall_index[0]]) +')',fontweight= 'bold',fontsize=15)
ax.set_ylabel(r'Predicted Target Value',fontsize=15)
ax.set_xlabel('Actual Target Value',fontsize=15)
ax.axis('equal')
plt.gca().set_aspect('equal', adjustable='box')
# ax.legend(loc="best")
# handles, labels = ax.get_legend_handles_labels()
# lgd = ax.legend(handles, labels, loc='center left',bbox_to_anchor=(1.05, 0.5)) 
plt.tight_layout()
plt.savefig("Predicted_vs_original_traning"+".pdf", bbox_inches='tight')
plt.show()