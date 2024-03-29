import os
import subprocess
import urllib.request

import numpy as np


def sum_to(x, shape):
    """
    - Reduce 연산이라서 x.ndim이 len(shape)보다 클 것이다.
    - x.shape의 원소는 shape의 원소보다 크거나 같을 것이다.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))
    
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.
    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.
    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]  # axis may contain negative values
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape  # scalar이거나 keepdims일 때

    gy = gy.reshape(shape)  # reshape
    return gy


def get_dot_graph(output, verbose=True):
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
            
    def _dot_var(v, verbose=False):
        dot_var = '{} [label="{}", color=orange, style=filled]\n'

        name = '' if v.name is None else v.name
        if verbose and v.data is not None:
            if v.name is not None:
                name += ': '
            name += str(v.shape) + ' ' + str(v.dtype)
        return dot_var.format(id(v), name)

    def _dot_func(f):
        dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
        txt = dot_func.format(id(f), f.__class__.__name__)

        dot_edge = '{} -> {}\n'
        for x in f.inputs:
            txt += dot_edge.format(id(x), id(f))
        for y in f.outputs:
            txt += dot_edge.format(id(f), id(y()))  # y는 약한 참조
        return txt

    txt = ''
    funcs = []
    seen_set = set()
    add_func(output.creator)
    txt += _dot_var(output, verbose)
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # 1. dot 데이터 저장
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')  # 사용자의 홈 디렉터리를 뜻하는 '~'의 절대 경로
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # 2. dot 명령 호출
    extension = os.path.splitext(to_file)[1][1:]  # 확장자
    cmd = f'dot {graph_path} -T {extension} -o {to_file}'
    subprocess.run(cmd, shell=True)
    
    # Return the image as a Jupyter Image object, to be displayed in-line.
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass


def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    y = np.exp(y)
    s = y.sum(axis=axis, keepdims=True)
    s = np.log(s)
    m += s
    return m

# =============================================================================
# download function
# =============================================================================
cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + '.' * (30 - i)
    print(bar_template.format(bar, p), end='')


def get_file(url, file_name=None):
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + 2 * pad - kernel_size) // stride + 1


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis, )
    else:
        axis = axis
    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape