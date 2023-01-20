# fx_graph_converter
Quick tool to extract FX Graph Modules from PyTorch networks and convert them into NetworkX graphs.

## How to use the tool
1. Use patch file to update PyTorch source. Make sure you have the right permissions to update PyTorch library.
```
patch <path_to_torch>/torch/_functorch/aot_autograd.py < fx_graph_converter_patch
```
2. Add call to TorchDynamo compilation in your model script. Example -
```
torch.compile(net, backend='nvprims_nvfuser')
```
I am using `nvprims_nvfuser` backend here to get FX Graph with higher level semantics. You can use TorchInductor (default backend) and you'll get a graph with lower level ops. For example, Softmax would be broken down into primitive ops.
Though I don't anticipate any issues, this code has not been tested with TorchInductor.
3. Install pydot package
```
pip install pydot
```

## Why edit PyTorch source when FX Graph can be obtained in model script?

## How to create your patch file for a different PyTorch source
Insert code from patch file available here into `torch/_functorch/aot_autograd.py` where joint FX Graph is available. Ideally look for 
```
if config.debug_joint:
```
Also insert the FX Graph convert counter at the top of the file with other `AOT_COUNTER` instantiation.

To create patch file, use
```
diff -u <path_to_original_torch>/torch/_functorch/aot_autograd.py <path_to_updated_aot_autograd>/aot_autograd.py > fx_graph_converter_patch
```

## What should you expect to see?
