is_simple_core = False

if is_simple_core:
    from dezero.core_simple import (Function, Variable, as_array, as_variable,
                                    no_grad, setup_variable, using_config)

else:
    import dezero.datasets
    import dezero.functions
    from dezero.core import (Function, Parameter, Variable, as_array,
                             as_variable, no_grad, setup_variable,
                             using_config)
    from dezero.layers import Layer
    from dezero.models import Model

setup_variable()
