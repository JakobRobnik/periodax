import numpy as np
import os
import quadpy



dirr = 'e3r2/'
# os.mkdir(dirr)

for order in range(7, 8):
    scheme = quadpy.e3r2.get_good_scheme(order)

    np.save(dirr + 'points' + str(order) + '.npy', scheme.points)
    np.save(dirr + 'weights' + str(order) + '.npy', scheme.weights)



# schemes in d-dimensions

# order = 5

# for dim in range(4, 10):

#     dirr = 'e'+str(dim)+'r2/'
#     #os.mkdir(dirr)
#     scheme = quadpy.enr2.stroud_enr2_5_2(dim)

#     np.save(dirr + 'points' + str(order) + '.npy', scheme.points)
#     np.save(dirr + 'weights' + str(order) + '.npy', scheme.weights)


