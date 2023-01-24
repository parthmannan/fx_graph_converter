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
4. Run your script as you normally would
5. Find the generated graph files in your run directory (networkx graph object pickle and generated graph drawings). More info below.

## Why edit PyTorch source when FX Graph can be obtained in model script?
There are a few ways to get the FX Graph of your network
1. Use symbolic trace
```
from torch.fx import symbolic_trace
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
```
This is not sufficient for us because a) It does not provide the backward pass graph and b) It 'symbolically' traces the computation graph without concrete input data, i.e. stores ops but does not store the tensor shapes etc.
Using `concrete_args` also doesn't lead to tensor information being propagated everywhere as expected.

2. Use make_fx
```
from torch.fx.experimental.proxy_tensor import make_fx
gm = make_fx(net)(x)
```
This is a much better option and comes close to meeting our requirements. However, this also does not generate the backward graph by default. One can insert an appropriate `torch.autograd.grad()` call in the network module to ensure backward graph gets traced but this requires tweaking the model scripts which may not be desirable.

3. Find an appropriate point in the PyTorch codebase
We are choosing this point `https://github.com/pytorch/pytorch/blob/7f2b5ea1e1495a1706ccf88269a0e920354240e3/torch/_functorch/aot_autograd.py#L1540`
Another possible option is in the partitioner `https://github.com/pytorch/pytorch/blob/master/torch/_functorch/partitioners.py#L319` but this has already modified the graph a little to suit the backend.


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
1. NetworkX Object pickle
The converter generates a networkx graph object in a `.pickle` format that can be used outside and independently of PyTorch to run graph analysis, manipulation etc.
To use the graph, add the following in a python script
```
nx_g = pickle.load(open('fx_graph_extracted_id_count(1).pickle', 'rb'))
```
Once you have the graph, you can inspect the nodes and edges.
Nodes:
1. Input nodes - Nodes starting with 'primals' are input nodes to the graph. These can be input data, biases, weights etc.
2. Backward inputs - Nodes starting with 'tangents' are input gradients to the backward graph.
3. Output nodes - Nodes with 'output' is the last sink of the graph and doesn't serve any purpose for the computation.
4. Regular nodes - All other nodes

Nodes have an attribute `node_call_info` which contains the `args` for that op. The script tries to map args provided by the FX Graph to the signature of the op itself (incl. default argument values etc.). In most cases, this should work but it is not thoroughly tested and not all signatures may be easily accessible.

Edges:
Edges are of (source, target, key) format where source=starting node, target=destination node and key=id of connection between them (0 for single connection)
Each edge has two important attributes `shape` and `dtype`. We can also add `memory_format` in the future.

An example to print out the tensor shapes of all tensors while running BFS on the graph once you have loaded the pickle file.
```
source_nodes = [x for x in nx_g.nodes if 'primals' in x]
source_nodes += [x for x in nx_g.nodes if 'tangents' in x]
edge_shapes = [nx_g.edges[x]['shape'] for x in nx.edge_bfs(nx_g, source=source_nodes)]
```

2. Graph drawing of the extracted graph
Example below
![Example Graph](examples/fx_graph_extracted_id_count(1).png)
