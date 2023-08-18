# cython-numpy

The associated youtube video is [here](https://youtu.be/zMctivubeIg)

## testing and timing

Open up an iPython section
```
import numpy as np
import pyximport
pyximport.install()

import inv
import cinv

x =np.random.rand(1000,3)

%timeit inv.invert(x, x[0])
%timeit inv.invert_numba(x, x[0])
%timeit cinv.invert1(x, x[0])
%timeit cinv.invert2(x, x[0])
```

## notes
- make code run correctly first, then add the compiler directives
- there are discussions on the preferred array allocation method in this [stackoverflow question](https://stackoverflow.com/questions/18462785/what-is-the-recommended-way-of-allocating-memory-for-a-typed-memory-view)
