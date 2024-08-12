from encoding import *
from eda_3 import *
import time

tic = time.time()
Scat.sctplot(df)
tac = time.time()
print(f"Time Taken {1000*(tac-tic)} ms")