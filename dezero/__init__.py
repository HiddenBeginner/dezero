is_simple_core = False

if is_simple_core:
    from dezero.core_simple import (Function, Variable, as_array, as_variable,
                                    no_grad, setup_variable, using_config)

else:
    import dezero.datasets
    import dezero.functions
    import dezero.transforms
    from dezero.core import (Config, Function, Parameter, Variable, as_array,
                             as_variable, no_grad, setup_variable, test_mode,
                             using_config)
    from dezero.dataloaders import DataLoader
    from dezero.layers import Layer
    from dezero.models import Model

import dezero.cuda
import dezero.optimizers

setup_variable()
