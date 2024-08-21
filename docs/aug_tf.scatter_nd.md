<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.scatter_nd" />
<meta itemprop="path" content="Stable" />
</div>

# tf.scatter_nd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">

Scatters `updates` into a tensor of shape `shape` according to `indices`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.manip.scatter_nd`, `tf.compat.v1.scatter_nd`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.scatter_nd(
    indices: Annotated[Any, TV_ScatterNd_Tindices],
    updates: Annotated[Any, TV_ScatterNd_T],
    shape: Annotated[Any, TV_ScatterNd_Tindices],
    name=None
) -> Annotated[Any, TV_ScatterNd_T]
</code></pre>



<!-- Placeholder for "Used in" -->

Scatter sparse `updates` according to individual values at the specified
`indices`. This op returns an output tensor with the `shape` you specify. This
op is the inverse of the <a href="../tf/gather_nd.md"><code>tf.gather_nd</code></a> operator which extracts values or slices
from a given tensor.

This operation is similar to <a href="../tf/tensor_scatter_nd_add.md"><code>tf.tensor_scatter_nd_add</code></a>, except that the tensor
is zero-initialized. Calling <a href="../tf/scatter_nd.md"><code>tf.scatter_nd(indices, updates, shape)</code></a>
is identical to calling
`tf.tensor_scatter_nd_add(tf.zeros(shape, updates.dtype), indices, updates)`

If `indices` contains duplicates, the associated `updates` are accumulated
(summed) into the output tensor.

**WARNING**: For floating-point data types, the output may be nondeterministic.
This is because the order in which the updates are applied is nondeterministic
and when floating-point numbers are added in different orders the resulting
numerical approximation error can be slightly different. However, the output
will be deterministic if op determinism is enabled via
<a href="../tf/config/experimental/enable_op_determinism.md"><code>tf.config.experimental.enable_op_determinism</code></a>.

`indices` is an integer tensor containing indices into the output tensor. The
last dimension of `indices` can be at most the rank of `shape`:

    indices.shape[-1] <= shape.rank

The last dimension of `indices` corresponds to indices of elements
(if `indices.shape[-1] = shape.rank`) or slices
(if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
`shape`.

`updates` is a tensor with shape:

    indices.shape[:-1] + shape[indices.shape[-1]:]

The simplest form of the scatter op is to insert individual elements in
a tensor by index. Consider an example where you want to insert 4 scattered
elements in a rank-1 tensor with 8 elements.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
</div>

In Python, this scatter operation would look like this:

```python
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
```

The resulting tensor would look like this:

    [0, 11, 0, 10, 9, 0, 0, 12]

You can also insert entire slices of a higher rank tensor all at once. For
example, you can insert two slices in the first dimension of a rank-3 tensor
with two matrices of new values.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
</div>

In Python, this scatter operation would look like this:

```python
    indices = tf.constant([[1], [3]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
```

The resulting tensor would look like this:

    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`indices`<a id="indices"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
Tensor of indices.
</td>
</tr><tr>
<td>
`updates`<a id="updates"></a>
</td>
<td>
A `Tensor`. Values to scatter into the output tensor.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
A `Tensor`. Must have the same type as `indices`.
1-D. The shape of the output tensor.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `updates`.
</td>
</tr>

</table>


</div>

<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target"></span>
**Issue:** How to use `tf.scatter_nd` to interleave columns of two 2D tensors in TensorFlow?

### Explanation
For 1D tensors, the interleaving can be achieved by specifying the indices where elements from each tensor should be placed and then using `tf.scatter_nd` to create the final interleaved tensor. This approach works well for 1D tensors, as demonstrated in the provided example where elements from two 1D tensors are interleaved to form a new tensor.

When extending this to 2D tensors, the challenge lies in specifying the correct indices for the columns rather than rows. The error encountered in the provided example arises because the indices are interpreted as row indices, leading to a mismatch in dimensions. To interleave columns, one must explicitly specify the indices for each element in the updates tensor, which can be cumbersome and error-prone.

A more scalable solution involves using slicing and concatenation. By slicing the tensors into the desired column groups and concatenating them along the appropriate axis, one can achieve the interleaving without manually specifying each index. This method is more efficient and easier to scale for tensors with a large number of columns.

### Code Example
This example demonstrates how to interleave columns of two 2D tensors using TensorFlow. It uses `tf.concat` to concatenate slices of the tensors along the column axis. This approach is scalable and avoids the complexity of manually specifying indices for `tf.scatter_nd`.
```python
import tensorflow as tf

# Define the input tensors
arr1 = tf.constant([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
arr2 = tf.constant([[10, 11, 12], [10, 11, 12], [10, 11, 12]])

# Interleave the columns of arr1 and arr2
result = tf.concat([arr1[:, :2], arr2[:, :1], arr1[:, 2:4], arr2[:, 1:2], arr1[:, 4:6], arr2[:, 2:3]], axis=1)

print(result.numpy()) # .numpy() converts the tensor to a numpy array for printing
```

### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/56969703/how-to-use-tf-scatter-nd-with-multi-dimensional-tensors" target="_blank">How to use `tf.scatter_nd` with multi-dimensional tensors</a>

- <a href="https://stackoverflow.com/questions/52572275/tensorflow-how-to-interleave-columns-of-two-tensors-e-g-using-tf-scatter-nd" target="_blank">tensorflow: how to interleave columns of two tensors (e.g. using tf.scatter_nd)?</a>

- <a href="https://stackoverflow.com/questions/56491633/what-is-the-difference-between-tf-scatter-add-and-tf-scatter-nd-when-indices-is" target="_blank">What is the difference between tf.scatter_add and tf.scatter_nd when indices is a matrix?</a>


</div>


