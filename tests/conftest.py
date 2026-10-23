import os
import sys

# The source lives in code/, which is not a package (it would clash with the stdlib 'code' module)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
