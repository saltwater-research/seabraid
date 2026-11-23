# Input capabilities

Seabraid accepts routed activation rows (FP32, or FP16 and BF16 for a provider that declares them),
expert IDs, and router weights. It returns FP32 partial contributions for the engine to merge. The library
never converts a 16-bit row: its bits reach the provider unchanged, and an undeclared format is refused
([F16 and BF16 activations](ADAPTER-GUIDE.md#f16-and-bf16-activations)). The same
interface can accept rows produced from text, image, audio, video, and other
inputs when the engine has encoded and routed them for a supported provider.
A mixed batch can carry multiple origin flags.

Use `seabraid_query_capabilities` or `Runtime.capabilities()` to discover the
operation, buffer formats, and source flags. The buffer formats it reports are the built-in F32 path; it cannot
advertise a provider's 16-bit support, which the engine knows from the flags it set on the provider. In C, set `out->abi_version` and
`out->struct_size` before the call; there is no separate version argument. These describe the ABI's input
contract. Model and device support come from the installed provider and
its qualified operation and weight formats.

```python
from seabraid import Runtime, SourceModality

with Runtime(library_path) as runtime:
    origins = SourceModality.AUDIO | SourceModality.VIDEO
    if runtime.capabilities().source_modalities & origins != origins:
        raise RuntimeError("This library does not accept these origins")
    operation = runtime.prepare(hidden=hidden, experts=experts, max_tokens=batch, top_k=k)
    partial = operation.weave(
        activations, expert_ids, router_weights, source_modalities=origins,
    )
    status = runtime.status()
```

The engine owns the media decoder, encoder, router, batching, and output
generation. The origin flags identify prepared activations. Arithmetic and placement
use the same shape and routing contract for every origin. Capabilities
and status are local API queries. An engine can relay them through its own
service endpoints as needed.

Qwen3-Omni gives us a model-level candidate for these routed expert stages.
A dense video diffusion denoiser requires a separate prepared operation and
provider with its own shape, model, and performance qualification.
[Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) ·
[SGLang Diffusion](https://sgl-project.github.io/diffusion/)
