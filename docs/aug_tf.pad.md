<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.pad" />
<meta itemprop="path" content="Stable" />
</div>

# tf.pad

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">


Pads a tensor.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.pad(
    tensor, paddings, mode=&#x27;CONSTANT&#x27;, constant_values=0, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This operation pads a `tensor` according to the `paddings` you specify.
`paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of
`tensor`. For each dimension D of `input`, `paddings[D, 0]` indicates how
many values to add before the contents of `tensor` in that dimension, and
`paddings[D, 1]` indicates how many values to add after the contents of
`tensor` in that dimension. If `mode` is "REFLECT" then both `paddings[D, 0]`
and `paddings[D, 1]` must be no greater than `tensor.dim_size(D) - 1`. If
`mode` is "SYMMETRIC" then both `paddings[D, 0]` and `paddings[D, 1]` must be
no greater than `tensor.dim_size(D)`.

The padded size of each dimension D of the output is:

`paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`

#### For example:



```python
t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1,], [2, 2]])
# 'constant_values' is 0.
# rank of 't' is 2.
tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
                                 #  [0, 0, 1, 2, 3, 0, 0],
                                 #  [0, 0, 4, 5, 6, 0, 0],
                                 #  [0, 0, 0, 0, 0, 0, 0]]

tf.pad(t, paddings, "REFLECT")  # [[6, 5, 4, 5, 6, 5, 4],
                                #  [3, 2, 1, 2, 3, 2, 1],
                                #  [6, 5, 4, 5, 6, 5, 4],
                                #  [3, 2, 1, 2, 3, 2, 1]]

tf.pad(t, paddings, "SYMMETRIC")  # [[2, 1, 1, 2, 3, 3, 2],
                                  #  [2, 1, 1, 2, 3, 3, 2],
                                  #  [5, 4, 4, 5, 6, 6, 5],
                                  #  [5, 4, 4, 5, 6, 6, 5]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`<a id="tensor"></a>
</td>
<td>
A `Tensor`.
</td>
</tr><tr>
<td>
`paddings`<a id="paddings"></a>
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`mode`<a id="mode"></a>
</td>
<td>
One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
</td>
</tr><tr>
<td>
`constant_values`<a id="constant_values"></a>
</td>
<td>
In "CONSTANT" mode, the scalar pad value to use. Must be
same type as `tensor`.
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
A `Tensor`. Has the same type as `tensor`.
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
When mode is not one of "CONSTANT", "REFLECT", or "SYMMETRIC".
</td>
</tr>
</table>

</div>

<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target"></span>
**Issue:** How do you use `tf.pad` to pad a 1-dimensional vector in TensorFlow?

### Explanation
To pad a 1-dimensional vector in TensorFlow, you need to use the `tf.pad` function correctly by specifying the `paddings` argument as a matrix with 2 columns. The `paddings` argument should be an integer tensor with shape `[n, 2]`, where `n` is the rank of the tensor you are padding. For each dimension `D` of the input tensor, `paddings[D, 0]` indicates how many values to add before the contents of the tensor in that dimension, and `paddings[D, 1]` indicates how many values to add after the contents of the tensor in that dimension.

To achieve the desired padding, you need to calculate the appropriate values for the `paddings` matrix. For example, if your original tensor has a length of 4 and you want to pad it to a length of 20, you need to add 16 values in total. You can distribute these values before and after the tensor as needed. Ensure that the `constant_values` argument is set to the value you want to pad with, which in this case is 10. The `mode` argument should be set to "CONSTANT" to use constant padding.


### Code Example
This example demonstrates how to pad a 1-dimensional tensor in TensorFlow to a specified length using the `tf.pad` function.
```python
import tensorflow as tf

# Define the 1-dimensional tensor
tensor = tf.constant([1, 2, 3, 45])

# Calculate the padding needed to reach the desired length of 20
padding_length = 20 - tf.shape(tensor)[0]

# Define the paddings matrix
paddings = [[0, padding_length]]

# Pad the tensor with constant values of 10
padded_tensor = tf.pad(tensor, paddings=paddings, constant_values=10)

# Print the result directly (no session needed in TF 2.x)
print(padded_tensor.numpy())  # Use .numpy() to get the values as a NumPy array
```

### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/42334646/tensorflow-pad-unknown-size-tensor-to-a-specific-size" target="_blank">TensorFlow - Pad unknown size tensor to a specific size?</a>

- <a href="https://stackoverflow.com/questions/43928642/how-does-tensorflow-pad-work" target="_blank">How does tensorflow.pad work?</a>

### Related YouTube Tutorials
Video tutorials to learn concepts related to the [**issue**](#target) with this API.
- <a href="https://www.youtube.com/watch?v=9ieVC_ABDNQ" target="_blank">TensorFlow Tutorial 18: TensorFlow Padding</a>

</div>


