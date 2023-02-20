is_simple_core = True

if is_simple_core:
    from dezero.core_simple import (Function, Variable, as_array, as_variable,
                                    no_grad, setup_variable, using_config)

else:
    pass

setup_variable()
