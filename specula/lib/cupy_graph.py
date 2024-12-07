

def cupy_graph(input_list=[], output_list=[], constant_list=[], synchronize=True):
    '''
    Decorator to capture a cupy function into a CUDA graph

    Parameters
    ----------
    input_list: list of parameter names with inputs that will point to
                different CuPY arrays at each graph run
    output_list: list of parameter names with outputs that will point to
                 different CuPY arrays at each graph run
    constant_list: list of parameter names with outputs that will point to
                 the same CuPY arrays at each graph run

    A static copy will be allocated for each parameter in the input
    and output lists. Input parameters are copied into the static copy
    before launching the graph, and output parameters are copied when done
    '''
    import cupy as cp
    import inspect
    import functools

    @functools.cache
    def func_args(f):
        '''List of parameter names for function *f*'''
        return inspect.getfullargspec(f).args

    def get_arg(f, parname, args, kwargs):
        '''Return a reference to the parameter *parname* of function *f*
        given the current arguments *args* and *kwargs*
        '''
        if parname in kwargs:
            return kwargs[parname]
        pos_args = func_args(f)
        if parname in pos_args:
            return args[pos_args.index(parname)]
        raise ValueError(f'Argument {argname} not found in function {func}')

    def decorator(f):

        def wrapper(*args, **kwargs):

            def copy_outputs(arg):
                for parname in decorator.outputs:
                    get_arg(f, parname, args, kwargs)[:] = decorator.outputs[parname][:]

            if hasattr(decorator, 'inputs') is False:

                # Static allocation of input/output parameters
                decorator.inputs = {}
                decorator.outputs = {}
                decorator.constants = {}
                for parname in input_list:
                    decorator.inputs[parname] = cp.empty_like(get_arg(f, parname, args, kwargs))

                for parname in output_list:
                    decorator.outputs[parname] = cp.empty_like(get_arg(f, parname, args, kwargs))

                for parname in constant_list:
                    decorator.constants[parname] = get_arg(f, parname, args, kwargs)

                # First run to capture graph
                decorator.stream = cp.cuda.Stream(non_blocking=False)
                with decorator.stream:
                    decorator.stream.begin_capture()
                    if func_args(f)[0] == 'self':
                        f(args[0], **{ **decorator.inputs, **decorator.outputs, **decorator.constants} )
                    else:
                        f( **{ **decorator.inputs, **decorator.outputs, **decorator.constants} )
                    decorator.cuda_graph = decorator.stream.end_capture()

            # Subsequent runs, copy inputs before running and copy outputs after
            for parname in decorator.inputs:
                decorator.inputs[parname][:] = get_arg(f, parname, args, kwargs)
            decorator.cuda_graph.launch(stream=decorator.stream)
            if synchronize:
                decorator.stream.synchronize()
                copy_outputs(None)
            else:
                decorator.stream.launch_host_func(callback=copy_outputs, arg=None)
        return wrapper
    return decorator
