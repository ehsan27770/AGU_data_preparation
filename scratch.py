import concurrent.futures
import itertools

def do_something(a,b,c,d):
    print(a + b + c + d)
    return 0

def do_nothing(a):
    print('done')
    return 0

if __name__ == '__main__':
    a = ['a','b','c','d']
    c = ['w','x','y','z']

    # %%
    with concurrent.futures.ProcessPoolExecutor() as executer:
        res = executer.map(do_something,a,itertools.repeat('ehsan'),c,itertools.repeat('-'))

    #with concurrent.futures.ProcessPoolExecutor() as executer:
    #    res = executer.map(do_nothing,a)
