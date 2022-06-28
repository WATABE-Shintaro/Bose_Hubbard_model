import numpy as np
cimport numpy as np
cimport cython

ctypedef  np.int64_t TYPE 

cdef class test:

    def __init__(self):
        pass

    def hhh(self,n, m):
        cdef np.ndarray[TYPE, ndim=2] aaa
        aaa = self.bbb(n, 1, m)
        return aaa


    cdef np.ndarray[TYPE, ndim=2] bbb(self,int n, int nt, int mt):
        cdef int i
        cdef np.ndarray[TYPE, ndim=2] ddd
        cdef np.ndarray[TYPE, ndim=2] fff
        cdef np.ndarray[TYPE, ndim=2] ccc
        if nt < n:
            ddd = self.bbb(n, nt + 1, mt)
            fff = np.full((len(ddd), 1), 0,dtype=np.int64)
            ccc = np.hstack([ddd, fff])
            for i in range(1, mt + 1):
                ddd = self.bbb(n, nt + 1, mt - i)
                fff = np.full((len(ddd), 1), i,dtype=np.int64)
                ddd = np.hstack([ddd, fff])
                ccc = np.vstack([ccc, ddd])
            return ccc
        else:
            ccc = np.full((1, 1), mt,dtype=np.int64)
            return ccc
