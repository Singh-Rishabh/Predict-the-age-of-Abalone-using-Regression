# ================== Import Statement ===================

import numpy as np 
import statistics
import random
from copy import deepcopy
from math import exp
from math import log
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as path_effects
from matplotlib import rcParams
import sys

# some default changes .....
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
		print (i)
	print(len(arr))

# function to convert an m*n matrix to n*m matrix
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


# function to calculate mean
def mean(l):
	return sum(l)/len(l)

# function to partition dataset.
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


# function to create dataset.

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
			output_arr.append(int(temp_output))
			
			temp_arr = list(map(float, temp_arr))
			data_arr.append(temp_arr)

			line = fp.readline()
			num_dataset = num_dataset + 1
			
	return data_arr,output_arr

# function to standardize_dataset
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


# function to calcuate norm 
def norm(w):
	sum = 0
	for i in range(len(w)):
		sum += w[i]*w[i]
	return sum

# function to calcualte error
def calc_error(X,Y,w,tmp_lambda):
	tmp_sum = 0
	tmp_f_of_x = X.dot(w)
	
	for i in range(len(X)):
		if (tmp_f_of_x[i] > 10) :
			tmp_f_of_x[i] = 10
		elif (tmp_f_of_x[i] < -10) :
			tmp_f_of_x[i] = -10
	
		tmp = 1.0/(1+ exp(-tmp_f_of_x[i]))
		if (tmp == 0 ):
			tmp = 0.0000001
		elif (tmp == 1):
			tmp = 0.999999
		tmp_sum = tmp_sum + Y[i]*log(tmp,10) + (1-Y[i])*log(1-tmp,10)
		
	# print(tmp_sum/len(X) + tmp_lambda*norm(w))
	return tmp_sum/len(X) + tmp_lambda*norm(w)

# function to calculate the descent
def calc_gradient_desent(X,Y,w,index):
	tmp_sum = 0
	tmp_f_of_x = X.dot(w)

	for i in range(len(X)):
		# print(tmp_f_of_x[i])
		if (tmp_f_of_x[i] > 10) :
			tmp_f_of_x[i] = 10
		elif (tmp_f_of_x[i] < -10) :
			tmp_f_of_x[i] = -10

		tmp = 1.0/(1+ exp(-tmp_f_of_x[i]))
		tmp_sum = tmp_sum + (tmp - Y[i])*X[i][index]
		
	return tmp_sum 

# function to calcuate netown rapson update
def calc_newton_rapson(X,Y,w,index,tmp_lambda):
	ideintity_matrix = []
	for i in range(len(X[0])):
		tmp_arr = []
		for j in range(len(X[0])):
			if (i==j):
				tmp_arr.append(1)
			else :
				tmp_arr.append(0)
		ideintity_matrix.append(tmp_arr)

	ideintity_matrix = np.asarray(ideintity_matrix)

	tmp_sum = 0
	tmp_f_of_x = X.dot(w)
	f_of_x = []
	R = []
	for i in range(len(X)):
		if (tmp_f_of_x[i] > 10) :
			tmp_f_of_x[i] = 10
		elif (tmp_f_of_x[i] < -10) :
			tmp_f_of_x[i] = -10

		tmp = 1.0/(1+ exp(-tmp_f_of_x[i]))
		f_of_x.append(tmp)
		tmp_arr = []
		for j in range(len(X)):
			if (i==j):
				tmp_arr.append(tmp*(1-tmp))
			else :
				tmp_arr.append(0)
		R.append(tmp_arr)

	out_arr = (np.matmul(np.linalg.pinv(np.matmul(np.matmul(np.transpose(X),R), X ) + tmp_lambda*ideintity_matrix ),np.transpose(X))).dot(f_of_x-Y) + tmp_lambda*w
	return out_arr

# function to perform logistic regression.
def mylogistic_regression_nr(X,Y,tmp_lambda,num_itter):
	alpha = 0.0001
	count = 1
	w = []
	if (len(X) != 0):
		for i in range(len(X[0])):
			w.append(0)
	else :
		print("Invalid Dataset... Exiting !!!!")
		exit()

	w = np.asarray(w, dtype = float)

	tmp_error = calc_error(X,Y,w,tmp_lambda)
	tmp_arr_w = calc_newton_rapson(X,Y,w,i,tmp_lambda)
	for i in range(len(w)):
		w[i] = w[i] - (tmp_arr_w[i] )
		
	new_error = calc_error(X,Y,w,tmp_lambda)
	
	while(abs(new_error - tmp_error) >= 0.000001 and count < num_itter):
		tmp_error = new_error
		tmp_arr_w = calc_newton_rapson(X,Y,w,i,tmp_lambda)
		for i in range(len(w)):
			w[i] = w[i] - (tmp_arr_w[i] )

		new_error = calc_error(X,Y,w,tmp_lambda)
		count += 1
	
	# print("Newton Rapson:::: Number of Itterations = " + str(count))
	return w


# function to perform gradient descent.
def mylogistic_regression_gd(X,Y,tmp_lambda,num_itter):
	alpha = 0.00001
	count = 1
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
		w[i] = w[i] - alpha*(calc_gradient_desent(X,Y,w,i) + tmp_lambda*2*w[i])
		
	new_error = calc_error(X,Y,w,tmp_lambda)
	
	# print(new_error,tmp_error),
	
	while(abs(new_error - tmp_error) >= 0.0000001 and count < num_itter):
		tmp_error = new_error
		# print(count),
		for i in range(len(w)):
			# print ("w is "+ str(w))
			w[i] = w[i] - alpha*(calc_gradient_desent(X,Y,w,i) + tmp_lambda*2*w[i])
		new_error = calc_error(X,Y,w,tmp_lambda)
		count += 1

	
	# print("Gradient Descent:::: Number of Itterations = " + str(count))
	return w

# function to calculte error.
def error_caclulation(X,Y,w):
	tmp_arr = X.dot(w)
	f_of_x = []
	for i in range(len(tmp_arr)):
		if (tmp_arr[i] > 10) :
			tmp_arr[i] = 10
		elif (tmp_arr[i] < -10) :
			tmp_arr[i] = -10

		f_of_x.append(1.0/(1+ exp(-tmp_arr[i])))

	return meansquarederr(f_of_x,Y)


# a utility function to cacluate error.
def meansquarederr(T, Tdash):
	sum_out = 0
	if (len(T) != len(Tdash)):
		print("output len missmatch error Exiting !!!!!")
		exit()
	for i in range(len(T)):
		# if ( (T[i] >=0.5 and Tdash[i] == 1) or (T[i] < 0.5 and Tdash[i] == 0) ):
			# sum_out += 1
		sum_out += (T[i] - Tdash[i])*(T[i] - Tdash[i])

	# return sum_out*100.0/len(T)
	return float(sum_out)/len(T)

# function to calculte accuracy.
def error_caclulation_acc(X,Y,w):
	tmp_arr = X.dot(w)
	# print(Y)
	f_of_x = []
	for i in range(len(tmp_arr)):
		if (tmp_arr[i] > 10) :
			tmp_arr[i] = 10
		elif (tmp_arr[i] < -10) :
			tmp_arr[i] = -10

		f_of_x.append(1.0/(1+ exp(-tmp_arr[i])))

	return meansquarederr_acc(f_of_x,Y)


# a utility function to cacluate accuracy.
def meansquarederr_acc(T, Tdash):
	sum_out = 0
	if (len(T) != len(Tdash)):
		print("output len missmatch error Exiting !!!!!")
		exit()
	for i in range(len(T)):
		# print("*******svscskjdckjcjbkc")
		# print(T[i] , Tdash[i])
		if ( (T[i] >=0.5 and Tdash[i] == 1) or (T[i] < 0.5 and Tdash[i] == 0) ):
			sum_out += 1
		# sum_out += (T[i] - Tdash[i])*(T[i] - Tdash[i])

	return sum_out*100.0/len(T)
	# return float(sum_out)/len(T)


# function to perform feature transform.
def featuretransform(X, degree):
	new_dataset = []
	for k in range(len(X)):
		tmp_arr = []
		for i in range(degree+1):
			for j in range(degree+1):
				if (i+j > degree):
					continue
				else :
					# print(X[k][0], X[k][1])
					tmp_arr.append((X[k][1]**i)*(X[k][2]**j))
		new_dataset.append(tmp_arr)
	return new_dataset




# call Read data function to make the dataset.
filepath = "./l2/credit.txt"
data_arr,output_arr = Read_data(filepath)

data_arr_backup = deepcopy(data_arr)
output_arr_backup = deepcopy(output_arr)


# partitioning the dataset
traning_dataset,traning_output,test_dataset,test_output = partition_dataset(data_arr,output_arr,1)

# standardizing dataset
traning_dataset = convertMat(traning_dataset)
traning_dataset,mean_arr,sd_arr = standardize_dataset(traning_dataset)
traning_dataset = convertMat(traning_dataset)

for k in range(len(traning_dataset)):
	traning_dataset[k].insert(0,1)
	
for k in range(len(test_dataset)):
	test_dataset[k].insert(0,1)

# converting the array to np array
traning_dataset = np.asarray(traning_dataset, dtype = float)
traning_output = np.asarray(traning_output, dtype = float)

test_dataset = np.asarray(test_dataset, dtype = float)
test_output = np.asarray(test_output, dtype = float)


# ******************************************

# making lambda array to perform logistic regression on variable number of lambda's
n = 50
lambda_Arr = [float(x + 1)/n for x in range(n)]
lambda_Arr.insert(0,0.0001)
lambda_Arr.insert(1,0.001)
# variation of error with different lambda

print ("\n*******************************************\n")
print("\nPrinting the effect of Lmabda on Error. The values \nof lambda are ")
for i in lambda_Arr:
	print(i),
print("\n")

error_vs_lamba_arr_nr= []
error_vs_lamba_arr_gd = []
for j in range(len(lambda_Arr)):
	# print("\nlambda equal " + str(lambda_Arr[j]))
	tmp_arr = mylogistic_regression_nr(traning_dataset,traning_output,lambda_Arr[j],1000)
	tmp_arr2 = mylogistic_regression_gd(traning_dataset,traning_output,lambda_Arr[j],1000)

	error_vs_lamba_arr_nr.append(error_caclulation(traning_dataset,traning_output,tmp_arr))
	error_vs_lamba_arr_gd.append(error_caclulation(traning_dataset,traning_output,tmp_arr2))

print("Error vs lambda (Newton Rapson)")
for i in error_vs_lamba_arr_nr:
	print (i),
print("\n")
print("\nError vs lambda (Gradiend Descent")

for i in error_vs_lamba_arr_gd:
	print (i),
print("\n")

# ********************************************
# variation of error with number of itteration.

error_arr_nr = []
error_arr_gd = []
num_itter_arr = [1,5,10,15,20,25,30,35,40,45,50,60,70,80,80,90,100,500,1000,10000]

print ("\n*******************************************\n")
print ("\nPrinting Effect of Number of itteration on Error.\nThe value of number of Itterations are folllowing")

for i in num_itter_arr:
	print(i),
print("\n")

for i in range(len(num_itter_arr)):
	# print("maximum itteration value = " + str(num_itter_arr[i]))
	tmp_arr = mylogistic_regression_nr(traning_dataset,traning_output,0.5,num_itter_arr[i])
	tmp_arr2 = mylogistic_regression_gd(traning_dataset,traning_output,0.5,num_itter_arr[i])

	error_arr_nr.append(error_caclulation(traning_dataset,traning_output,tmp_arr))
	error_arr_gd.append(error_caclulation(traning_dataset,traning_output,tmp_arr2))

print("\n")
print("Error vs Numnber of Itterations (Newton Rapson")

for i in error_arr_nr:
	print (i),
print("\n")

print("\nError vs Numnber of Itterations (Gradient Descent)")


for i in error_arr_gd:
	print (i),
print("\n")

# ********************************************

# calculating error wrt degree
print ("\n*******************************************\n")
print("\nLogistic regression after basis function.\nThe value of Degree of polynomial are following")
error_arr_nr_deg = []
error_arr_gd_deg = []

error_arr_nr_deg_acc = []
error_arr_gd_deg_acc = []
degree_arr = [1,2,3,4,5,6,7,8,9,10]

for i in degree_arr:
	print (i),
print("\n")

w_arr_nr = []
w_arr_gd = []
for i in range(len(degree_arr)):

	new_dataset = featuretransform(traning_dataset,degree_arr[i])
	new_dataset = np.asarray(new_dataset)

	tmp_arr = mylogistic_regression_nr(new_dataset,traning_output,0.5,1000)
	tmp_arr2 = mylogistic_regression_gd(new_dataset,traning_output,0.5,1000)

	w_arr_gd.append(tmp_arr2)
	w_arr_nr.append(tmp_arr)

	error_arr_nr_deg.append(error_caclulation(new_dataset,traning_output,tmp_arr))
	error_arr_gd_deg.append(error_caclulation(new_dataset,traning_output,tmp_arr2))

	error_arr_nr_deg_acc.append(error_caclulation_acc(new_dataset,traning_output,tmp_arr))
	error_arr_gd_deg_acc.append(error_caclulation_acc(new_dataset,traning_output,tmp_arr2))


# print ("\n*******************************************\n")
# print("\nLogistic regression after basis function for diferent lambda ")
error_arr_nr_4_lambda = []
error_arr_gd_4_lambda = []
for i in range(len(lambda_Arr)):
	# print("\nlambda equal " + str(lambda_Arr[i]))
	new_dataset = featuretransform(traning_dataset,4)
	new_dataset = np.asarray(new_dataset)


	tmp_arr = mylogistic_regression_nr(new_dataset,traning_output,lambda_Arr[i],1000)
	tmp_arr2 = mylogistic_regression_gd(new_dataset,traning_output,lambda_Arr[i],1000)

	error_arr_nr_4_lambda.append(error_caclulation(new_dataset,traning_output,tmp_arr))
	error_arr_gd_4_lambda.append(error_caclulation(new_dataset,traning_output,tmp_arr2))


# ********************************************************************
# ********************** Plots ***************************************
print("\n")
print(error_arr_nr_deg)
print(error_arr_gd_deg)

max_accuracy_nr = 2
max_acciracy_gd = 2
for i in range(len(error_arr_nr_deg)):
	if (error_arr_nr_deg[max_accuracy_nr-2]  < error_arr_nr_deg[i]):
		max_accuracy_nr = i+2
	if (error_arr_gd_deg[max_acciracy_gd-2]  < error_arr_gd_deg[i]):
		max_acciracy_gd = i+2
print(max_accuracy_nr,max_accuracy_nr)


# ***************************
# Dataset Plot
tmp_arr = convertMat(data_arr)
arr1 = []
arr2 = []
arr3 = []
arr4 = []
 
for i in range(len(tmp_arr[0])):
	if (output_arr[i] == 1):
		arr1.append(tmp_arr[0][i])
		arr2.append(tmp_arr[1][i])
	else :
		arr3.append(tmp_arr[0][i])
		arr4.append(tmp_arr[1][i])

fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
		
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.scatter(arr1,arr2,label = "Issued", marker = "o")
ax.scatter(arr3,arr4,label = "Rejected",marker = "^")
ax.set_xlabel("Attribute 1",fontsize=15)
ax.set_ylabel("Attribute 2",fontsize=15)
ax.legend(loc="best")
ax.set_title('Dataset ',fontweight= 'bold',fontsize=15)
plt.tight_layout()
plt.savefig("logistic_regression_dataser"+".pdf", bbox_inches='tight')

# **********Error vs Lambda**********

fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
		
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.plot(lambda_Arr,error_vs_lamba_arr_gd,label = "Gradient Descent",color = 'C1')
ax.plot(lambda_Arr,error_vs_lamba_arr_nr,label = "Newton Rapson",color = 'C2')
ax.set_xlabel(r"$\lambda$",fontsize=15)
ax.set_ylabel("Error",fontsize=15)
ax.set_title('Error vs Lambda',fontweight= 'bold',fontsize=15)

ax.legend(loc="best")
plt.tight_layout()
plt.savefig("nr_vs_gd_on_lambda"+".pdf", bbox_inches='tight')
# plt.show()


# **********Error vs Number of Itterations************
fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
		
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.plot(num_itter_arr, error_arr_gd, label = "Gradient Descent",color = 'C1')
ax.plot(num_itter_arr, error_arr_nr, label = "Newton Rapson",color = 'C2')
ax.set_xlabel(r"Number of itteration",fontsize=15)
ax.set_ylabel("Error",fontsize=15)
ax.set_title('Error vs Number of Itterations',fontweight= 'bold',fontsize=15)
ax.legend(loc="best")

plt.tight_layout()
plt.savefig("nr_vs_gd_on_numItter"+".pdf", bbox_inches='tight')
# plt.show()

# **********Error vs Degree of polynomial************
fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
		
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.plot(degree_arr,error_arr_gd_deg,label = "Gradient Descent",color = 'C1')
ax.plot(degree_arr,error_arr_nr_deg,label = "Newton Rapson",color = 'C2')
ax.set_xlabel(r"Degree",fontsize=15)
ax.set_ylabel("Error",fontsize=15)
ax.legend(loc="best")
ax.set_title('Error vs Degree of Polynimial',fontweight= 'bold',fontsize=15)
plt.tight_layout()
plt.savefig("nr_vs_gd_on_degree"+".pdf", bbox_inches='tight')
# plt.show()


# **********accuracy vs Degree of polynomial************
fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
		
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.plot(degree_arr,error_arr_gd_deg_acc,label = "Gradient Descent",color = 'C1')
ax.plot(degree_arr,error_arr_nr_deg_acc,label = "Newton Rapson",color = 'C2')
ax.set_xlabel(r"Degree",fontsize=15)
ax.set_ylabel("Accuracy",fontsize=15)
ax.legend(loc="best")
ax.set_title('Error vs Degree of Polynimial',fontweight= 'bold',fontsize=15)
plt.tight_layout()
plt.savefig("nr_vs_gd_on_degree_Acc"+".pdf", bbox_inches='tight')
# plt.show()


# *********Error vs Degree 4 Polynomial*************
fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
		
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.plot(lambda_Arr,error_arr_gd_4_lambda,label = "Gradient Descent",color = 'C1')
ax.plot(lambda_Arr,error_arr_nr_4_lambda,label = "Newton Rapson",color = 'C2')
ax.set_xlabel(r"$\lambda$",fontsize=15)
ax.set_ylabel("Error",fontsize=15)
ax.legend(loc="best")
ax.set_title(r'Error vs $4^{th}$ Degree polynomial',fontweight= 'bold',fontsize=15)
plt.tight_layout()
plt.savefig("nr_vs_gd_on_4degree_lambda"+".pdf", bbox_inches='tight')
# plt.show()

# =================================


print_arr(w_arr_nr)


for k in range(len(1)):
	x_min, x_max = 0 , 6
	y_min, y_max = 0 , 6
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
	# print("\n\n")
	# print(xx)
	# print(yy)
	z = np.zeros(shape=np.shape(xx), dtype=float)
	fig, ax = plt.subplots()
	ax.yaxis.grid(True)
	ax.set_axisbelow(True)
			
	ax.minorticks_on()
	ax.tick_params(axis='x',which='minor',bottom='off')
	ax.yaxis.grid(True)
	ax.set_axisbelow(True)
	ax.scatter(arr1,arr2,label = "Issued", marker = "o")
	ax.scatter(arr3,arr4,label = "Rejected",marker = "^")
	ax.set_xlabel("Attribute 1",fontsize=15)
	ax.set_ylabel("Attribute 2",fontsize=15)
	ax.legend(loc="best")
	ax.set_title('Dataset ',fontweight= 'bold',fontsize=15)
	for i in range(np.shape(xx)[0]):
	    for j in range(np.shape(xx)[1]):

	        mat = featuretransform([[1, xx[i][j], yy[i][j] ]], degree_arr[k])
	        mat = np.asarray(mat)
	        res = mat.dot(w_arr_nr[k])

	        z[i][j] = res[0]

	plt.contour(xx,yy,z, 0)
	plt.savefig("1stdegree_polynomial" + str(k+1))

plt.show()