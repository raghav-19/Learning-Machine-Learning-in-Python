import numpy as np

a=np.array([4,2,3])
b=np.array([[4,5,1],[7,2,9],[2,7,8]])

print(a) # to print array
a.ndim   # return number of dimension in array
a.shape  # return tuple which is size of array
a.dtype  # return data type of array
a.size   # return number of element in array
a.itemsize # return size of each element in array

c=np.array([1,2,3],dtype='int8') # this is how we can assign initial data type to array
r=np.arange(20) # return 1D array of element [0,20)
rs=r.reshape(2,2,5) # return array after reshape total number of element should be same
b[1,1] # return (r,c) element in array
b[0:2,1:3] # return range of element

zero=np.zeros((2,3))
one=np.ones((3,2))
full=np.full((3,3),10)
full_like=np.full_like(a,45)
random=np.random.rand(4,2)
a_random= np.random.random_sample(a.shape)
r_rand=np.random.randint(-4,8,size=(3,3))
r[2:10:3] # return array (start:end:step)
arr=np.repeat(a,3,axis=1) # if axis is 1 it will add in same row, if axis is 0 then it will add in same column
x=a.copy() # return new array same as a


"""
operation on matrix
b=a+2 # new array with all element +2
b=a*2 # new array with all element *2
b=a**2 # new array with all element ^2
c=a+b # return added array dimension must be same
c=np.matmul(a,b) # return multiplicative array, must be multiplicable
b=np.amin(a)
b=np.amin(a,0)
b=np.amin(a,1)
d=np.linalg.det(b) # determinant of array

d=np.vstack([a,a,a]) it will make 3 rows of a extend number of rows
d=np.hstack([a,a,a]) it will make 9 column of a extend number of column
"""

# reading writing in file
filedata=np.genfromtxt('finaldata.txt',delimiter=',')
np.save('outfile',b) # file extension .npy
z=np.load('infile.npy')

np.savetxt('outfile.txt',a)
z=np.loadtxt('infile.txt')


# Boolean Masking and Advance Indexing
b>3 # return boolean array of condition
y=b[b>3] # return 1D array with element on true condition
y=np.any(b>3,axis=1) # it axis=0,1 acc to condition
y=np.all(b>3,axis=1) # return 1D boolean array with condition






