import numpy as np


# 1D array
lst1 = [1,2,3]
arr1 = np.array(lst1)
#print(lst)
#print(arr)


# 2D array 
lst2 = [[1,2,3],[4,5,6],[7,8,9]]
arr2 = np.array(lst2)
#print(lst1)
#print(arr1)


# arange function
#print(np.arange(0,10)) #Creates an array with numbers from 0-9
#print(np.arange(0,11,2)) #Creates an array with numbers from 0-10 with step 2


# generate an array of all zeros
#print(np.zeros(3))
#print(np.zeros((5,4))) # 1st number represents rows and 2nd number represent columns


# generate an array of all ones
#print(np.ones(3))
#print(np.ones((5,4))) # 1st number represents rows and 2nd number represent columns


# linspace - accepts 3 numbers: 1-start, 2-end, 3-number of evenly spaced numbers
#print(np.linspace(1,10,10))


# to create n-dimentional identity matrix
#print(np.eye(4))


# to create an array of random uniform numbers 0-1
#print(np.random.rand(5)) # will create 1D array with 5 random numbers from 0-1
#print(np.random.rand(5,5)) # will create 2D array with 5X5 random numbers from 0-1


# to return samples from std normal/gaussian distribution cetenred around 0
#print(np.random.randn(2)) # will create 1D array with 2 random numbers centered around 0
#print(np.random.rand(4,3)) # will create 2D array with 4X2 random numbers centered around 0


# to create an array of random interger
#print(np.random.randint(1,60)) # this will return only a single number
#print(np.random.randint(1,60,10)) # this will return 10 random integers from 1-60



arr3 = np.arange(25)
ranarr = np.random.randint(0,50,10)

# to find out the shape of the array
#print(arr3.shape)

# reshape function
#print(arr3.reshape(5,5)) # will reshape the 1D array to 5X5 array

# to find the max or min values/positions
#print(ranarr.max()) # this will return max value from the array
#print(ranarr.min()) # this will return min value from the array
#print(ranarr.argmax()) # this will return max value's position from the array
#print(ranarr.argmin()) # this will return min value's position from the array

#print(arr3.dtype) # to find out the data type of the array


################## Numpy Indexing and Selection ###############

arr4 = np.arange(0,11)

#print(arr4[4]) # Indexing is similar to python list
#print(arr4[2:5])
#print(arr4[1:])
#print(arr4[:5])

#arr4[:5] = 100 # this will assign the positions from 0-4 with value 100

# on changing the sliced part original array will get modified
#arr4_slice = arr4[:6]
#arr4_slice[:] = 22
#print(arr4)
#print(arr4_slice)


# .cpoy() method will prevent modification of the original array
#arr_copy = arr4.copy()
#arr_copy[:] = 99
#print(arr4)
#print(arr_copy)


# creating a 2D array with Numpy
arr_2D = np.array([[5,10,15],[20,25,30],[35,40,45]])
#print(arr_2D[2][1]) # this a double bracket method to grab an element
#print(arr_2D[2,1]) # this a single bracket method to grab an element
#print(arr_2D[:2,1:])


# Boolian operator 
arr5 = np.arange(1,11)

#print(arr5 > 5) # will return a list of Ture/False

#lst3 = arr5[arr5 > 5]   # this will create a list of values where the condition holds true
#print(lst3)


arr_2D = np.arange(50).reshape(5,10) #creating 2D array with arange and reshape func.
#print(arr_2D)
#print(arr_2D[1:3])
#print(arr_2D[1:3,3:5])



################## Numpy Operations ################

# Array with Array operations

arr6 = np.arange(0,11)
#print(arr6 + arr6) # this will add elements of two arrays wrt its positions
#print(arr6 - arr6)  #           substract
#print(arr6 * arr6)  #           multiply
#print(arr6 / arr6)  #           divide


# Array with Scalar operations

#print(arr6 + 100) # this will add elements of two arrays wrt its positions
#print(arr6 - 1)  #           substract
#print(arr6 * 2)  #           multiply
#print(arr6 / 2)  #           divide


# Universal Array Functions

#print(np.sqrt(arr6)) # this will make square root of each of its elements
#print(np.exp(arr6))  # this will give the exponential value of each of its elements
#print(np.max(arr6))  # max value
#print(arr6.max())    # max value
#print(np.min(arr6))  # min value
#print(np.sin(arr6))  # sin values
#print(np.log(arr6))  # log values
#print(np.sum(arr6))  # sum of all values
#print(arr_2D.sum(axis=0)) # sum of all the columns
#print(np.std(arr6))  # standard deviation

## these are some basic ones but there are lot more.....



