# Quantizing Models with Core AI Composite Ops in Graph Mode

Core AI recognizes certain well-known building blocks, such as SDPA or RMSNorm, as _composite ops_ and applies optimized implementations for them at runtime. These composite ops are available via [coreai_torch.composite_ops](https://apple.github.io/coreai-torch/main/api/composite-ops.html). When a PyTorch model uses one of these ops it can be converted using one of the following two APIs: 

- **Using `TorchConverter.add_pytorch_module()` along with the `externalize_modules` arg**: As described [here](https://apple.github.io/coreai-torch/main/guides/composite-ops.html), this approach takes as input an `nn.Module` torch model, along with the composite op torch module, specified via the *[externalize_modules](https://apple.github.io/coreai-torch/main/guides/externalization.html)* arg. Any coreai-opt optimizer that yields a torch model of type `nn.Module` can be converted via this route. 
- **Using `TorchConverter.add_exported_program()`** :  This [coreai-torch API](https://apple.github.io/coreai-torch/main/api/TorchConverter.html#add-exported-program) operates on an already exported torch program. When using coreai-opt's quantizer with [graph execution mode](../quantization/overview.md#two-execution-modes-graph-and-eager), which results in a torch `ExportedProgram`, there are a few additional steps required to ensure  correct conversion and externalization of the composite op modules. This guide describes this process via an example. 



## Example 

This example uses the same `RMSNormComposite` op from the [Externalization](https://apple.github.io/coreai-torch/main/guides/externalization.html) guide, however, the same process applies for all composite ops with their respective `ExternalizeSpec`s.


First define the model: 

```python
import torch
import torch.nn as nn

# The composite op
class RMSNormComposite(nn.Module):
    def __init__(self, axes=-1, eps=1e-5, version=1):
        super().__init__()
        self.axes = axes
        self.eps = eps
        self.version = version

    def forward(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_f32 = input.to(torch.float32)
        inv_rms = torch.rsqrt((x_f32 * x_f32).mean(self.axes, keepdim=True) + self.eps)
        return (input * inv_rms).to(input.dtype) * scale


# A model that uses the composite op
class Model(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = RMSNormComposite()
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        return self.out(self.norm(self.proj(x), self.norm_weight))


original_model = Model().eval()
example_inputs = (torch.randn(1, 32),)
```

Now apply Quantization with graph mode, and convert using coreai-torch. 

```python
import coreai_opt as opt
from coreai_opt.quantization import Quantizer, QuantizerConfig
from coreai_opt.quantization.config import ExecutionMode
import coreai_torch
from coreai_torch import ExternalizeSpec, TorchConverter, _patch_model_for_externalization, _subexport_and_restore


# Patch the model in-place
# to mark RMSNormComposite module for "externalization" 
_patch_model_for_externalization(
    original_model,
    targets=[
        ExternalizeSpec(
            target_class=RMSNormComposite,
            composite_op_name="rms_norm",
            composite_attrs=["axes", "eps", "version"],
        )
    ],
)

# Usual coreai-opt APIs for quantization
config = QuantizerConfig.presets.w8(execution_mode=ExecutionMode.GRAPH)
quantizer = Quantizer(original_model, config)
prepared_model = quantizer.prepare(example_inputs)
finalized_quantized_model = quantizer.finalize(backend=opt.ExportBackend.CoreAI)

# Export to Core AI
exported_program = torch.export.export(finalized_quantized_model, example_inputs).run_decompositions(
    coreai_torch.get_decomp_table()
)
composite_module_exported_programs = _subexport_and_restore(original_model, exported_program) 
coreai_program = (
    TorchConverter()
    .add_exported_program(
        exported_program, _externalized_exported_programs=composite_module_exported_programs
    )
    .to_coreai()
)
```

Compared to the [usual flow](../introduction/integration_coreai.md), there are *two* key differences: 

- **Call to `_patch_model_for_externalization`**: this is made prior to invoking any coreai-opt API. This API modifies the original torch model in place. It replaces the forward method of the `nn.module`s specified in `ExternalizeSpec(target_class)` with a `torch.library.custom_op` that invokes the original forward method. This keeps the model functionally the same, however, now the sub-module (RMSNormComposite in this example) becomes opaque to the `torch.export.export` process: a single node. In graph mode execution, coreai-opt's `quantizer.prepare` invokes the torch.export API. Hence no quantization will be applied *inside* the sub-module. A few additional information/metadata is also registered and saved as attributes on the sub-module of interest, which is used by the later APIs.  
- **Call to `_subexport_and_restore` and passing the returned object to the `add_exported_program` API**: After coreai-opt APIs are used (either for PTQ or QAT) and the model is exported as usual, call to this method is made with the *original* starting torch model. Internally this API, walks the original model, identifies the sub-modules, and `torch.export`s each instance and returns them in a list. These are then used by the `add_exported_program` API, to prepare the coreai graph (`aimodel`), which contains both the main graph and separately preserved sub-graphs (pattern matched) corresponding to the composite op submodule.  

While the insides of the composite module cannot be quantized, its boundaries (incoming and outgoing tensors) can be, using the usual `module_input_spec` and `module_output_spec` config kwargs. 

:::{warning} The externalization APIs used below, _patch_model_for_externalization and _subexport_and_restore in coreai-torch are currently experimental and hence prefixed with an underscore. :::