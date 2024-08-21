<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.conv1d" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.conv1d

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">

Computes a 1-D convolution given 3-D input and filter tensors.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.conv1d(
    input,
    filters,
    stride,
    padding,
    data_format=&#x27;NWC&#x27;,
    dilations=None,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given an input tensor of shape
  `batch_shape + [in_width, in_channels]`
if `data_format` is `"NWC"`, or
  `batch_shape + [in_channels, in_width]`
if `data_format` is `"NCW"`,
and a filter / kernel tensor of shape
`[filter_width, in_channels, out_channels]`, this op reshapes
the arguments to pass them to `conv2d` to perform the equivalent
convolution operation.

Internally, this op reshapes the input tensors and invokes <a href="../../tf/nn/conv2d.md"><code>tf.nn.conv2d</code></a>.
For example, if `data_format` does not start with `"NC"`, a tensor of shape
  `batch_shape + [in_width, in_channels]`
is reshaped to
  `batch_shape + [1, in_width, in_channels]`,
and the filter is reshaped to
  `[1, filter_width, in_channels, out_channels]`.
The result is then reshaped back to
  `batch_shape + [out_width, out_channels]`
\(where out_width is a function of the stride and padding as in conv2d\) and
returned to the caller.

<!-- Tabular view -->
<table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A Tensor of rank at least 3. Must be of type `float16`, `float32`, or
`float64`.
</td>
</tr><tr>
<td>
`filters`<a id="filters"></a>
</td>
<td>
A Tensor of rank at least 3.  Must have the same type as `input`.
</td>
</tr><tr>
<td>
`stride`<a id="stride"></a>
</td>
<td>
An int or list of `ints` that has length `1` or `3`.  The number of
entries by which the filter is moved right at each step.
</td>
</tr><tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
'SAME' or 'VALID'. See
[here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
for more information.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
An optional `string` from `"NWC", "NCW"`.  Defaults to `"NWC"`,
the data is stored in the order of
`batch_shape + [in_width, in_channels]`.  The `"NCW"` format stores data
as `batch_shape + [in_channels, in_width]`.
</td>
</tr><tr>
<td>
`dilations`<a id="dilations"></a>
</td>
<td>
An int or list of `ints` that has length `1` or `3` which
defaults to 1. The dilation factor for each dimension of input. If set to
k > 1, there will be k-1 skipped cells between each filter element on that
dimension. Dilations in the batch and depth dimensions must be 1.
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
A `Tensor`.  Has the same type as input.
</td>
</tr>

</table>



<!-- Tabular view -->
<table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `data_format` is invalid.
</td>
</tr>
</table>

</div>

<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target"></span>
**Issue:** Why TensorFlow Uses 2D Convolutions for 1D Convolution Operations?

### Explanation
In TensorFlow, the `tf.nn.conv1d` function is designed to perform 1D convolution operations on 3D input tensors. However, internally, TensorFlow reshapes the input tensors and invokes the `tf.nn.conv2d` function to perform the equivalent convolution operation.

When `tf.nn.conv1d` is called, the input tensor of shape `[batch, in_width, in_channels]` (assuming the default data format `"NWC"`) is reshaped to `[batch, 1, in_width, in_channels]`. Similarly, the filter tensor of shape `[filter_width, in_channels, out_channels]` is reshaped to `[1, filter_width, in_channels, out_channels]`. This reshaping allows the 1D convolution to be treated as a special case of the 2D convolution, where one of the spatial dimensions has a size of 1. After the 2D convolution operation is performed, the result is reshaped back to `[batch, out_width, out_channels]`.

The primary reason for this approach is to leverage the existing optimized implementation of 2D convolutions in TensorFlow. By reusing the `tf.nn.conv2d` function, TensorFlow can avoid duplicating code and ensure that the 1D convolution benefits from the same performance optimizations and hardware acceleration as the 2D convolution.

### Code Example
Below code demonstrates how TensorFlow uses 2D convolutions internally for `tf.nn.conv1d` by reshaping inputs to leverage optimized 2D operations.
```python
import tensorflow as tf
import numpy as np
# Define a 1D input tensor with shape [batch, in_width, in_channels]
input_tensor = tf.random.normal([1, 10, 3])

# Define a 1D filter tensor with shape [filter_width, in_channels, out_channels]
filter_tensor = tf.random.normal([3, 3, 2])

# Perform 1D convolution using tf.nn.conv1d
conv1d_output = tf.nn.conv1d(input_tensor, filter_tensor, stride=1, padding='SAME')

# Print the shape of the output tensor
print('Output shape of 1D convolution:', conv1d_output.shape)

# Reshape the input and filter tensors to perform equivalent 2D convolution
input_tensor_2d = tf.reshape(input_tensor, [1, 1, 10, 3])
filter_tensor_2d = tf.reshape(filter_tensor, [1, 3, 3, 2])

# Perform 2D convolution using tf.nn.conv2d
conv2d_output = tf.nn.conv2d(input_tensor_2d, filter_tensor_2d, strides=[1, 1, 1, 1], padding='SAME')

# Reshape the 2D convolution output back to 1D
conv2d_output_reshaped = tf.reshape(conv2d_output, [1, 10, 2])

# Print the shape of the reshaped 2D convolution output
print('Output shape of reshaped 2D convolution:', conv2d_output_reshaped.shape)

# Verify that the outputs of 1D and reshaped 2D convolutions are the same
print('Are the outputs equal?', np.allclose(conv1d_output.numpy(), conv2d_output_reshaped.numpy()))

```
### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/71175695/tensorflow-conv1d-keras-conv1d-strange-performance-variation" target="_blank">Tensorflow conv1d/Keras Conv1D strange performance variation</a>


### Related YouTube Tutorials
Video tutorials to learn concepts related to the [**issue**](#target) with this API.
- <a href="https://www.youtube.com/watch?v=SNpRfTINJbA" target="_blank">89: conv1d | TensorFlow | Tutorial</a>

</div>


