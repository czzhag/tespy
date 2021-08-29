import numpy as np
from scipy.io import loadmat

def get_array_info(fdir):

	fd = loadmat(fdir)
	pnames = fd['p'].dtype.names
	indnames=fd['ind'].dtype.names

	class Struct:
		def __init__(self, nd, names):
			for n in names:
				exec("self.%s = nd['%s'][0,0]"%(n,n))

	p = Struct(fd['p'],pnames)
	ind = Struct(fd['ind'],indnames)

	return p,ind	

