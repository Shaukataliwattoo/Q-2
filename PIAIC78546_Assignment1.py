#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[2]:


import numpy as np


# 2. Create a null vector of size 10 

# In[6]:


null_vector = np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[7]:


arr = np.arange(10,50)
arr


# 4. Find the shape of previous array in question 3

# In[8]:


arr.shape


# 5. Print the type of the previous array in question 3

# In[9]:


type(arr)


# 6. Print the numpy version and the configuration
# 

# In[10]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[12]:


arr.ndim


# 8. Create a boolean array with all the True values

# In[14]:


arr = np.ones(10, dtype = bool)
arr


# 9. Create a two dimensional array
# 
# 
# 

# In[18]:


arr1 = np.arange(9).reshape(3,3)
arr1


# 10. Create a three dimensional array
# 
# 

# In[19]:


ndim3 = np.arange(12).reshape(2,2,3)
ndim3


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[21]:


arr[::-1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[23]:


nul_vector = np.zeros(10)
nul_vector[4] = 1
nul_vector


# 13. Create a 3x3 identity matrix

# In[21]:


identity_matrix = np.identity(3)
identity_matrix


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[26]:


arr = np.array([1,2,3,4,5])
arr = np.asfarray(arr)
arr.dtype


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[42]:


arr1 = np.array([[1., 2., 3.],
                 [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],
                 [7., 2., 12.]])
result = arr1*arr2
result


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[29]:


arr1 = np.array([[1., 2., 3.],
                 [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],
                 [7., 2., 12.]])
maximum = np.maximum(arr1,arr2)
maximum


# 17. Extract all odd numbers from arr with values(0-9)

# In[30]:


arr = np.arange(10)
arr = arr[arr%2 != 0]
arr


# 18. Replace all odd numbers to -1 from previous array

# In[33]:


arr = np.arange(10, dtype = float)
arr[arr%2 != 0] = -1
arr


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[72]:


arr = np.arange(10)
arr[5:9] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[104]:


arr = np.ones((4,4))
arr[1:3,1:3] = 0
arr


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[75]:


arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
arr2d[1,1] = 12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[80]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0] = 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first value of 1st 1-D array from it

# In[34]:


arr2d = np.arange(9).reshape(3,3)
arr2d[0,0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[87]:


arr2d = np.arange(9).reshape(3,3)
arr2d[1,1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[93]:


arr2d = np.arange(9).reshape(3,3)
arr2d[:-1,2]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[46]:


arr = np.random.randint(100, size = (10,10))
print('Maximum value is:', np.max(arr))
print('Minimum value is:', np.min(arr))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[54]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
c = np.intersect1d(a,b)
c


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[124]:


# a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# a[a == b]


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[55]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data[names != "Will"]


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[56]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data[(names != "Will") & (names != "Joe")]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[58]:


arr = np.random.randn(1,15).reshape(5,3)
arr


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[67]:


arr = np.random.randn(1,16).reshape(2,2,4)
arr


# 33. Swap axes of the array you created in Question 32

# In[68]:


arr.T


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[72]:


arr = np.arange(10)
arr = np.sqrt(arr)
np.where(arr<0.5,0,arr)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[76]:


arr = np.random.randint(12)
arr1 = np.random.randint(12)
maximum_vaues = np.maximum(arr, arr1)
maximum_vaues


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[77]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
unique_names = np.unique(names)
unique_names


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[ ]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[80]:


sampleArray = np.array([[34,43,73],
                        [82,22,12],
                        [53,94,66]])
newColumn = np.array([[10,10,10]])
sampleArray[:,1] = newColumn
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[81]:


x = np.array([[1., 2., 3.], 
              [4., 5., 6.]]) 
y = np.array([[6., 23.], 
              [-1, 7], 
              [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[90]:


arr = np.random.randint(20, size = (1,20))
np.cumsum(arr)

