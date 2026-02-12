

import numpy as np

vec1 = np.array([1,2,3])
vec2 = np.array([4,5,6])


print(vec1.shape) # should be (3,)


# we can do a dotproduct, as well as an outer prod
dotted = np.dot(vec1,vec2)
print(dotted)

vec1_matrix = vec1.reshape(3,1)
vec2_matrix = vec2.reshape(3,1)

# NP DOES NOT HAVE UNSQUEEZE 
vec1_matrix_expanded_dims = vec1_matrix[..., None]
print('vec1_matrix_expanded_dims shape', vec1_matrix_expanded_dims.shape)


# NP matrix mult:
np_matrix = np.eye(2)
np_matrix = np.zeros((2,2))
np_matrix[0,1] = 1
np_matrix[1,0] = 1
print(np_matrix)
vec_xy = np.array([1,2])
print(vec_xy, 'to', np_matrix@vec_xy)


# reshape vs view
print(np_matrix.flatten()) # -> [0 1 1 0 ]
# use reshape 99% of the time 
print(np_matrix.reshape((1,4)))

nparange = np.arange(10)
print(nparange)
print(nparange.shape)

print(vec1_matrix.shape)
# in order to make it 
outer = np.matmul(vec1_matrix, vec2_matrix.T)
print(outer)

