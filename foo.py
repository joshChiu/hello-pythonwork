print('Hello world')

import sys
# %timeit np.linalg.eigvals(np.random.rand(100,100))

# %%writefile [-a] myfile.py

# %%writefile -a test.py print('测试%%writefile魔法')
# print(a)


# %lsmagic
# # %%writefile [-a] foo.py

# # print('Hello world')

# # %run ipykernel_launcher.py

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--verbosity", help="increase output verbosity")
# args = parser.parse_args(args=[])
# print(args)

# # sys.argv.append('hello')


# print(sys.argv)


# a=sys.argv[0]
# b=sys.argv[1]


# print(a,b)

