<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.embedding_lookup_sparse" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.embedding_lookup_sparse

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/embedding_ops.py">View source</a>

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">

Looks up embeddings for the given ids and weights from a list of tensors.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    combiner=None,
    max_norm=None,
    name=None,
    allow_fast_lookup=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

`params` is a dense tensor or a list of dense tensors, and `sp_ids` is a 2D
<a href="../../tf/sparse/SparseTensor.md"><code>tf.SparseTensor</code></a> or <a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> indicating the indices of `params` to
gather.

This op is best described with an example. Suppose `params` is an embedding
table of size `(4, 2)` and `sp_ids` has 3 rows. Since `sp_ids` is sparse or
ragged, not every row has the same number of elements. The output has shape
(3, 2). Each row of `sp_ids` is a list of indices, where each index selects a
row of `params`. For a given row of `sp_ids`, the rows of `params` are
gathered based on the indices in `sp_ids`, then combined by taking their sum
or mean.

```
>>> params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
>>> sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
...                          values=[0, 1, 3, 2], dense_shape=(3, 2))
>>> tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights=None,
...                               combiner='sum').numpy()
array([[4., 6.], [7., 8.], [5., 6.]], dtype=float32)
```

In this example, `sp_ids` has 3 rows, so the output has 3 rows. Row 0 of
`sp_ids` has values 0 and 1, so it selects rows 0 and 1 from `params`, which
are `[1, 2]` and `[3, 4]`. The rows are summed since `combiner='sum'`,
resulting in the output row of `[4, 6]`.

Since row 1 and 2 of `sp_ids` only have one value each, they simply select the
corresponding row from `params` as the output row. Row 1 has value `3` so
it selects the `params` elements `[7, 8]` and row 2 has the value 2 so it
selects the `params` elements `[5, 6]`.

If `sparse_weights` is specified, it must have the same shape as `sp_ids`.
`sparse_weights` is used to assign a weight to each slice of `params`. For
example:

```
>>> params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
>>> sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
...                          values=[0, 1, 3, 2], dense_shape=(3, 2))
>>> sparse_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
...                                  values=[0.1, 1.0, 0.5, 2.0],
...                                  dense_shape=(3, 2))
>>> tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights=sparse_weights,
...                               combiner='sum').numpy()
array([[3.1, 4.2], [3.5, 4.], [10., 12.]], dtype=float32)
```

In general, `params` can have shape `(p0, ..., pn)` and `sp_ids` can have `M`
rows, where each row can have any number of elements. The output has shape
`(M, p1, ..., pn)`. Each slice of the output `output[i, ...]` is obtained as
follows: The `combiner` argument is used to combine the values
`params[sp_ids[i, j], ...] * sparse_weights[i, j]` for each `j` in `range(0,
len(sp_ids[i]))`, e.g. by taking the sum or mean of the values.

This op assumes that there is at least one id for each row in the dense tensor
represented by sp_ids (i.e. there are no rows with empty features), and that
all the indices of sp_ids are in canonical row-major order.

`sp_ids` and `sp_weights` (if not None) are `SparseTensor`s or `RaggedTensor`s
with rank of 2. For `SpareTensor`s with left-aligned non-zero entries which
can be described as `RaggedTensor`s, use of `RaggedTensor`s can yield higher
performance.

This op assumes that all id values lie in the range [0, p0), where p0
is `params.shape[0]`. If you want a version of this op that prunes id values
less than 0, see <a href="../../tf/nn/safe_embedding_lookup_sparse.md"><code>tf.nn.safe_embedding_lookup_sparse</code></a>

If `len(params) > 1`, each element of `sp_ids` is partitioned between the
elements of `params` according to the "div" partition strategy, which means we
assign ids to partitions in a contiguous manner. For instance, 13 ids are
split across 5 partitions as:
`[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

If the id space does not evenly divide the number of partitions, each of the
first `(max_id + 1) % len(params)` partitions will be assigned one more id.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`params`<a id="params"></a>
</td>
<td>
A single tensor representing the complete embedding tensor, or a
list of tensors all of same shape except for the first dimension,
representing sharded embedding tensors following "div" partition strategy.
</td>
</tr><tr>
<td>
`sp_ids`<a id="sp_ids"></a>
</td>
<td>
N x M `SparseTensor` of int64 ids where N is typically batch size
and M is arbitrary or a `RaggedTensor` with rank 2.
</td>
</tr><tr>
<td>
`sparse_weights`<a id="sparse_weights"></a>
</td>
<td>
`SparseTensor` or `RaggedTensor` of same type and shape as
`sparse_ids`, containing float / double weights corresponding to
`sparse_ids`, or `None` if all weights are assumed to be 1.0.
</td>
</tr><tr>
<td>
`combiner`<a id="combiner"></a>
</td>
<td>
A string specifying the reduction op. Currently "mean", "sqrtn"
and "sum" are supported. "sum" computes the weighted sum of the embedding
results for each row. "mean" is the weighted sum divided by the total
weight. "sqrtn" is the weighted sum divided by the square root of the sum
of the squares of the weights. Defaults to `mean`.
</td>
</tr><tr>
<td>
`max_norm`<a id="max_norm"></a>
</td>
<td>
If not `None`, each embedding is clipped if its l2-norm is larger
than this value, before combining.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional name for the op.
</td>
</tr><tr>
<td>
`allow_fast_lookup`<a id="allow_fast_lookup"></a>
</td>
<td>
An optional boolean specifying whether to allow
simplified embedding lookups when `params` is a single tensor and
`max_norm` is `None`. Setting this flag to `True` during training can
cause the use of dense gradients with increased memory footprint.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dense tensor representing the combined embeddings for the
sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
looks up the embeddings for all ids in that row, multiplies them by the
corresponding weight, and combines these embeddings as specified.

In other words, if

  `shape(combined params) = [p0, p1, ..., pm]`

and

  `shape(sp_ids) = shape(sp_weights) = [d0, d1]`

then

  `shape(output) = [d0, p1, ..., pm]`.

For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

  ```python
  [0, 0]: id 1, weight 2.0
  [0, 1]: id 3, weight 0.5
  [1, 0]: id 0, weight 1.0
  [2, 3]: id 1, weight 3.0
  ```

with `combiner`="mean", then the output will be a 3x20 matrix where

  ```python
  output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
  output[1, :] = (params[0, :] * 1.0) / 1.0
  output[2, :] = (params[1, :] * 3.0) / 3.0
  ```
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`<a id="TypeError"></a>
</td>
<td>
If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
neither `None` nor `SparseTensor`.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If `combiner` is not one of {"mean", "sqrtn", "sum"}.
</td>
</tr>
</table>

</div>

<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target"></span>
**Issue:** How to convert a 3D sparse tensor into a 2D sparse tensor for use with `tf.nn.embedding_lookup_sparse`, and then reshape the output back to 3D?

### Explanation
`tf.nn.embedding_lookup_sparse` is designed to look up embeddings for given ids and weights from a list of tensors. According to the TensorFlow API documentation, `sp_ids` must be a 2D `SparseTensor` or `RaggedTensor`. This means that `tf.nn.embedding_lookup_sparse` does not natively support 3D sparse tensors.

To achieve the desired output dimensions, the recommended approach is to convert the 3D sparse tensor into a 2D sparse tensor. This involves flattening the 3D structure into a 2D array, performing the embedding lookup, and then reshaping the result back into the desired 3D shape. By doing so, the embedding lookup can be performed correctly, and the final output can be reshaped to match the expected dimensions.

### Code Example
Below is a code example demonstrating how to convert a 3D sparse tensor to a 2D sparse tensor, perform embedding lookup with `tf.nn.embedding_lookup_sparse`, and then reshape the result to the desired shape.
```python
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import enable_eager_execution, get_variable
from tensorflow import SparseTensor, nn

# Enable eager execution
enable_eager_execution()
# Define the embedding matrix
w = get_variable("w", [4, 4], initializer=tf.random_normal_initializer())

# Original 3D array
z = np.array([
    [[0, 1, 2, 3], [2, 3]],
    [[1, 3], [2]],
    [[0, 1, 3], [1, 2]]
])

# Convert the 3D array to a 2D array
z_2d = np.array([
    [0, 1, 2, 3],
    [2, 3],
    [1, 3],
    [2],
    [0, 1, 3],
    [1, 2]
])

# Create a 2D sparse tensor
sp = SparseTensor(values=[0, 1, 2, 3, 2, 3, 1, 3, 2, 0, 1, 3, 1, 2],
                 indices=[[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[2,0],[2,1],[3,0],[4,0],[4,1],[4,2],[5,0],[5,1]],
                 dense_shape=[6, 4])

# Perform the embedding lookup
res = nn.embedding_lookup_sparse(w, sp, None, combiner='sum')

# Reshape the result to the desired shape
reshaped_res = tf.reshape(res, [-1, 2, 4])

# Print the results
print("Embedding matrix (w):")
print(w.numpy())
print("\nResult after embedding lookup and reshaping:")
print(reshaped_res.numpy())
```
### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/60013980/tf-nn-embedding-lookup-sparse-3d-sparse-tensor-input" target="_blank">tf.nn.embedding_lookup_sparse 3D sparse tensor input</a>

- <a href="https://stackoverflow.com/questions/39207587/how-to-use-tf-nn-embedding-lookup-sparse-in-tensorflow" target="_blank">How to use tf.nn.embedding_lookup_sparse in TensorFlow?</a>

- <a href="https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do" target="_blank">What does tf.nn.embedding_lookup function do?</a>

### Related YouTube Tutorials
Video tutorials to learn concepts related to the [**issue**](#target) with this API.
- <a href="https://www.youtube.com/watch?v=vbHX5R1rIo8" target="_blank">Accelerate embedding lookup operation (Building recommendation systems with TensorFlow)</a>

</div>


