import os
import random
import string
from sympy import Integer, sqrt, symbols, exp, sin, cos, Symbol
import sympy
from src.envs.char_sp import CharSPEnvironment
from src.utils import AttrDict
from multiprocessing import Pool

params = AttrDict({

    # Environment Parameters
    'env_name': 'char_sp',
    'int_base': 10,
    'balanced': False,
    'positive': True,
    'precision': 10,
    'n_variables': 1,
    'n_coefficients': 0,
    'leaf_probs': '0.75,0,0.25,0',
    'max_len': 512,
    'max_int': 5,
    'max_ops': 15,
    'max_ops_G': 15,
    'clean_prefix_expr': True,
    'rewrite_functions': '',
    'tasks': 'ode1',
    'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,'
                 'acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1',

    # Model Parameters
    'cpu': False,
    'emb_dim': 1024,
    'n_enc_layers': 6,
    'n_dec_layers': 6,
    'n_heads': 8,
    'dropout': 0,
    'attention_dropout': 0,
    'sinusoidal_embeddings': False,
    'share_inout_emb': True,
    'reload_model': 'ode2.pth',

})
env = CharSPEnvironment(params)

c1, c2 = symbols('a9 a8')
x = Symbol('x')

def random_string(length):
    letters_and_digits = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

digits = 2
filedir = f'dumped/ode2_data/{digits}digits/{random_string(10)}/'
os.makedirs(filedir)

def get_line(coffs):
    # randomly generate a, b, c
    a = Integer(coffs[0])
    b = Integer(coffs[1])
    c = Integer(coffs[2])

    # get y_prefix by a, b, c
    delta = b**2 - 4*a*c
    alpha = -b / (2*a) 
    beta = sqrt(abs(delta)) / (2*a)
    if delta > 0:
        y = c1 * exp((alpha+beta)*x) + c2 * exp((alpha-beta)*x) 
    elif delta == 0:
        y = c1 * exp(alpha*x) + c2 * x * exp(alpha*x)
    else:
        y = c1 * exp(alpha*x) * cos(beta*x) + c2 * exp(alpha*x) * sin(beta*x)
    y_prefix = env.sympy_to_prefix(y)
    # y_infix = env.prefix_to_infix(y_prefix)

    # get eq_prefix by a, b, c
    eq_infix = f'({a}) * Derivative(Derivative(f(x))) + ({b}) * Derivative(f(x)) + ({c}) * f(x)'
    eq = sympy.sympify(eq_infix, locals=env.local_dict)
    eq_prefix = env.sympy_to_prefix(eq)
    eq_prefix = env.clean_prefix(eq_prefix)
    # eq_infix = env.prefix_to_infix(eq_prefix)

    # get line by y_prefix, eq_prefix
    line = ' '.join(eq_prefix) + '\t' + ' '.join(y_prefix) + '\n'
    with open(filedir + 'data.prefix', mode='a') as f:
        f.write(line)



def generate_random_coffs(n=100, bound=99):
    for _ in range(n):
        a = 0
        while a == 0:
            a = random.randint(-bound, bound)
        b = random.randint(-bound, bound)
        c = random.randint(-bound, bound)
        yield (a, b, c)

if __name__ == '__main__':
    with Pool() as p:
        p.map(get_line, generate_random_coffs(n=10*(digits*3), bound=10**digits-1))