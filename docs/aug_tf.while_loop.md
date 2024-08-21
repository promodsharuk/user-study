<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.while_loop" />
<meta itemprop="path" content="Stable" />
</div>

# tf.while_loop

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/while_loop.py">View source</a>

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">

Repeat `body` while the condition `cond` is true. (deprecated argument values)


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.while_loop(
    cond,
    body,
    loop_vars,
    shape_invariants=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    maximum_iterations=None,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: SOME ARGUMENT VALUES ARE DEPRECATED: `(back_prop=False)`. They will be removed in a future version.
Instructions for updating:
back_prop=False is deprecated. Consider using tf.stop_gradient instead.
Instead of:
results = tf.while_loop(c, b, vars, back_prop=False)
Use:
results = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(c, b, vars))

Note: This op is automatically used in a <a href="../tf/function.md"><code>tf.function</code></a> to convert Python for-
and while- loops when the loop variable is a <a href="../tf/Tensor.md"><code>tf.Tensor</code></a>, unless
`autograph=False` is explicitly specified in <a href="../tf/function.md"><code>tf.function</code></a> args. For example,
the following are equivalent:

```
>>> @tf.function
... def sumSquare(n):
...   i, result = tf.constant(0), tf.constant(0)
...   while i < n: # AutoGraph converts while-loop to tf.while_loop().
...     result += i * i
...     i += 1
...   return result
>>> sumSquare(10).numpy()
285
```

```
>>> @tf.function
... def sumSquare2(n):
...   i, result = tf.constant(0), tf.constant(0)
...   c = lambda i, _: tf.less(i, n)
...   b = lambda i, result: (i + 1, result + i * i)
...   return tf.while_loop(c, b, [i, result])[1]
>>> sumSquare2(10).numpy()
285
```

For more information, see [tf.function and AutoGraph guide
](https://www.tensorflow.org/guide/function#autograph_transformations).

`cond` is a callable returning a boolean scalar tensor. `body` is a callable
returning a (possibly nested) tuple, namedtuple or list of tensors of the same
arity (length and structure) and types as `loop_vars`. `loop_vars` is a
(possibly nested) tuple, namedtuple or list of tensors that is passed to both
`cond` and `body`. `cond` and `body` both take as many arguments as there are
`loop_vars`.

In addition to regular Tensors or IndexedSlices, the body may accept and
return TensorArray objects.  The flows of the TensorArray objects will
be appropriately forwarded between loops and during gradient calculations.

Note that `while_loop` calls `cond` and `body` *exactly once* (inside the
call to `while_loop`, and not at all during `Session.run()`). `while_loop`
stitches together the graph fragments created during the `cond` and `body`
calls with some additional graph nodes to create the graph flow that
repeats `body` until `cond` returns false.

For correctness, <a href="../tf/while_loop.md"><code>tf.while_loop()</code></a> strictly enforces shape invariants for
the loop variables. A shape invariant is a (possibly partial) shape that
is unchanged across the iterations of the loop. An error will be raised
if the shape of a loop variable after an iteration is determined to be more
general than or incompatible with its shape invariant. For example, a shape
of `[11, None]` is more general than a shape of `[11, 17]`, and `[11, 21]` is
not compatible with `[11, 17]`. By default (if the argument `shape_invariants`
is not specified), it is assumed that the initial shape of each tensor in
`loop_vars` is the same in every iteration. The `shape_invariants` argument
allows the caller to specify a less specific shape invariant for each loop
variable, which is needed if the shape varies between iterations. The
<a href="../tf/Tensor.md#set_shape"><code>tf.Tensor.set_shape</code></a>
function may also be used in the `body` function to indicate that
the output loop variable has a particular shape. The shape invariant for
SparseTensor and IndexedSlices are treated specially as follows:

a) If a loop variable is a SparseTensor, the shape invariant must be
`TensorShape([r])` where `r` is the rank of the dense tensor represented
by the sparse tensor. It means the shapes of the three tensors of the
SparseTensor are `([None], [None, r], [r])`. NOTE: The shape invariant here
is the shape of the SparseTensor.dense_shape property. It must be the shape of
a vector.

b) If a loop variable is an IndexedSlices, the shape invariant must be
a shape invariant of the values tensor of the IndexedSlices. It means
the shapes of the three tensors of the IndexedSlices are `(shape, [shape[0]],
[shape.ndims])`.

`while_loop` implements non-strict semantics, enabling multiple iterations
to run in parallel. The maximum number of parallel iterations can be
controlled by `parallel_iterations`, which gives users some control over
memory consumption and execution order. For correct programs, `while_loop`
should return the same result for any `parallel_iterations > 0`.

For training, TensorFlow stores the tensors that are produced in the
forward inference and are needed in back propagation. These tensors are a
main source of memory consumption and often cause OOM errors when training
on GPUs. When the flag swap_memory is true, we swap out these tensors from
GPU to CPU. This for example allows us to train RNN models with very long
sequences and large batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cond`<a id="cond"></a>
</td>
<td>
A callable that represents the termination condition of the loop.
</td>
</tr><tr>
<td>
`body`<a id="body"></a>
</td>
<td>
A callable that represents the loop body.
</td>
</tr><tr>
<td>
`loop_vars`<a id="loop_vars"></a>
</td>
<td>
A (possibly nested) tuple, namedtuple or list of numpy array,
`Tensor`, and `TensorArray` objects.
</td>
</tr><tr>
<td>
`shape_invariants`<a id="shape_invariants"></a>
</td>
<td>
The shape invariants for the loop variables.
</td>
</tr><tr>
<td>
`parallel_iterations`<a id="parallel_iterations"></a>
</td>
<td>
The number of iterations allowed to run in parallel. It
must be a positive integer.
</td>
</tr><tr>
<td>
`back_prop`<a id="back_prop"></a>
</td>
<td>
(optional) Deprecated. False disables support for back
propagation. Prefer using <a href="../tf/stop_gradient.md"><code>tf.stop_gradient</code></a> instead.
</td>
</tr><tr>
<td>
`swap_memory`<a id="swap_memory"></a>
</td>
<td>
Whether GPU-CPU memory swap is enabled for this loop.
</td>
</tr><tr>
<td>
`maximum_iterations`<a id="maximum_iterations"></a>
</td>
<td>
Optional maximum number of iterations of the while loop
to run.  If provided, the `cond` output is AND-ed with an additional
condition ensuring the number of iterations executed is no greater than
`maximum_iterations`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional name prefix for the returned tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The output tensors for the loop variables after the loop. The return value
has the same structure as `loop_vars`.
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
if `cond` or `body` is not callable.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `loop_vars` is empty.
</td>
</tr>
</table>



#### Example:



```
>>> i = tf.constant(0)
>>> c = lambda i: tf.less(i, 10)
>>> b = lambda i: (tf.add(i, 1), )
>>> r = tf.while_loop(c, b, [i])[0]
>>> r.numpy()
10
```

Example with nesting and a namedtuple:

```
>>> import collections
>>> Pair = collections.namedtuple('Pair', 'j, k')
>>> ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
>>> c = lambda i, p: i < 10
>>> b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
>>> ijk_final = tf.while_loop(c, b, ijk_0)[1]
>>> ijk_final[0].numpy(), ijk_final[1].numpy()
(32, 64)
```

Example using shape_invariants:

```
>>> i0 = tf.constant(0)
>>> m0 = tf.ones([2, 2])
>>> c = lambda i, m: i < 10
>>> b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
>>> tf.while_loop(
...     c, b, loop_vars=[i0, m0],
...     shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])[1]
<tf.Tensor: shape=(2048, 2), dtype=float32, numpy=...>
```

Example which demonstrates non-strict semantics: In the following
example, the final value of `counter` does not depend on `x`. So
the `while_loop` can increment the counter parallel to updates of `x`.
However, because the loop counter at one loop iteration depends
on the value at the previous iteration, the loop counter itself cannot
be incremented in parallel. Hence if we just want the final value of the
counter (which we print on the line `print(sess.run(i))`), then
`x` will never be incremented, but the counter will be updated on a
single thread. Conversely, if we want the value of the output (which we
print on the line `print(sess.run(out).shape)`), then the counter may be
incremented on its own thread, while `x` can be incremented in
parallel on a separate thread. In the extreme case, it is conceivable
that the thread incrementing the counter runs until completion before
`x` is incremented even a single time. The only thing that can never
happen is that the thread updating `x` can never get ahead of the
counter thread because the thread incrementing `x` depends on the value
of the counter.

```
>>> with tf.compat.v1.Session() as sess:
...   n = 10
...   c = lambda i, x: i < n
...   b = lambda i, x: (
...       tf.compat.v1.Print(i + 1, [i], "Updating i based on i == "),
...       # Let x depend on i
...       tf.compat.v1.Print(x + i, [i], "Updating x based on i == "))
...
...   # Make x to be a big matrix so its updating thread would run slowly
...   x = tf.zeros([1000, 100], dtype=tf.int32)
...   counter = tf.constant(0)
...   counter_out, x_out = tf.while_loop(c, b, (counter, x))
...
...   # The following line may increment the counter and x in parallel.
...   # The counter thread may get ahead of the x thread, but not the
...   # other way around. For example, the log may contain these messages:
...   # ```
...   # Updating i based on i == [9]
...   # Updating x based on i == [3]
...   # ```
...   # meaning that the counter(i) thread is on iteration 9,
...   # while the x thread is on iteration 3.
...   print(sess.run(x_out).shape)
(1000, 100)
```



</div>

<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target"></span>
**Issue:** How to use `tf.while_loop` with eager execution in TensorFlow 2.0.

### Explanation
The `tf.while_loop` function requires the `body` argument to be a callable that returns a tuple of tensors. This callable is executed repeatedly until the `cond` callable returns `False`.

In the context of eager execution, which is the default mode in TensorFlow 2.0, the `body` function must explicitly return the updated loop variables. If the `body` function does not return the updated loop variables, TensorFlow will raise a `ValueError` because it expects the `body` to return a value that can be used in the next iteration of the loop.The issue arises when the `body` function is defined as a standard Python function that does not return the updated loop variables. In eager execution, TensorFlow attempts to convert the return value of the `body` function to a tensor, and if the return value is `None`, it results in a `ValueError`.

To resolve this, ensure that the `body` function explicitly returns the updated loop variables. This allows TensorFlow to correctly propagate the loop variables through each iteration of the loop. Additionally, the `cond` function should also be a callable that returns a boolean scalar tensor, which determines whether the loop should continue.

### Code Example
This example demonstrates how to use tf.while_loop with eager execution.
```python
import tensorflow as tf
# Define the condition function
c = lambda i: tf.less(i, 10)

# Define the body function
@tf.function
def b(i):
    return tf.add(i, 1),

# Initialize the loop variable
i = tf.constant(0)

# Execute the while loop
result = tf.while_loop(c, b, [i])

# Print the result
print(result[0].numpy())
```

### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/37441140/how-to-use-tf-while-loop-in-tensorflow" target="_blank">How to use tf.while_loop() in tensorflow</a>

- <a href="https://stackoverflow.com/questions/59332392/tf-function-and-tf-while-loop-in-tensorflow-2-0" target="_blank">tf.function and tf.while loop in Tensorflow 2.0</a>

- <a href="https://stackoverflow.com/questions/41604686/how-to-use-tf-while-loop-for-variable-length-inputs-in-tensorflow" target="_blank">How to use tf.while_loop() for variable-length inputs in tensorflow?</a>

- <a href="https://stackoverflow.com/questions/46768386/while-loop-error-in-tensorflow" target="_blank">while_loop error in Tensorflow</a>

### Related YouTube Tutorials
Video tutorials to learn concepts related to the [**issue**](#target) with this API.
- <a href="https://www.youtube.com/watch?v=KrS94hG4VU0" target="_blank">Custom Training Loops for Multi Headed Tensorflow 2 Models</a>

- <a href="https://www.youtube.com/watch?v=RPocW_aMZKE" target="_blank">Inside TensorFlow: TF Debugging</a>

</div>


<div style="padding: 5px; float: right; width: 34%; margin-left: 5px; height: 10px;"><br></div>



<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target"></span>
**Issue:** How do you implement a TensorFlow `tf.while_loop` with a condition dependent on a tensor computed within the loop body?

### Explanation
The `tf.while_loop` function allows for the creation of a loop in which the body of the loop and the condition for continuation are defined by separate callable functions. Both the condition (`cond`) and the body (`body`) functions must take the same arguments, which are provided in the `loop_vars`.

To perform random cropping on an image and continue this process until a crop of sufficient quality is obtained, the key is to ensure that the tensor representing the cropped image (`img_crop`) is included in the `loop_vars` so that it can be passed between the `body` and `cond` functions. The `body` function will perform the random cropping and return the updated `img_crop` along with any other loop variables. The `cond` function will then evaluate the quality of the `img_crop` and determine whether the loop should continue.

The `tf.while_loop` function will call the `cond` function with the initial values of `loop_vars` before entering the loop. After each iteration, the updated values of `loop_vars` returned by the `body` function are passed back to the `cond` function to check if the loop should continue. This process ensures that the condition can depend on the tensor computed within the loop body.

### Code Example
This example demonstrates how to use `tf.while_loop` to perform random cropping on an image until a crop of sufficient quality is obtained.
```python
import numpy as np
import tensorflow as tf

IMAGE_SHAPE = [960, 720]
CROP_SHAPE = [320, 240]
max_begin_index = np.array(IMAGE_SHAPE) - np.array(CROP_SHAPE)

# Here, we assume the image is fixed and use tf.constant
define_img = tf.constant(np.random.rand(960, 720, 3), dtype=tf.float32)  

# Initial value for img_crop
define_img_crop = tf.zeros([CROP_SHAPE[0], CROP_SHAPE[1], 3], dtype=tf.float32)

# Body function for the while loop
def body(image_shape, crop_shape, img_crop, img):
    crop_begin_index = tf.round(tf.random.uniform([2]) * max_begin_index)
    img_crop = tf.slice(img, crop_begin_index, crop_shape + [-1])
    return image_shape, crop_shape, img_crop, img

# Condition function for the while loop
def cond(image_shape, crop_shape, img_crop, img):
    return tf.cast(tf.math.count_nonzero(img_crop > 0), tf.int64) > tf.cast(0.5 * tf.cast(tf.size(img_crop), tf.float32), tf.int64)  

# Using tf.while_loop to perform the cropping
image_shape, crop_shape, img_crop, img = tf.while_loop(
    cond=cond,
    body=body,
    loop_vars=([960, 720], [320, 240], define_img_crop, define_img),
    shape_invariants=([None], [None], tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3]))
)

result = img_crop.numpy()  # Convert the result to a NumPy array
print(result)

```

### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/53032922/tensorflow-while-loop-with-condition-dependent-on-body" target="_blank">TensorFlow while loop with condition dependent on body</a>

- <a href="https://stackoverflow.com/questions/37441140/how-to-use-tf-while-loop-in-tensorflow" target="_blank">How to use tf.while_loop() in tensorflow</a>


### Related YouTube Tutorials
Video tutorials to learn concepts related to the [**issue**](#target) with this API.
- <a href="https://www.youtube.com/watch?v=306tG48lhhQ" target="_blank">23: Basic training loops | TensorFlow | Tutorial</a>

- <a href="https://www.youtube.com/watch?v=H_9dthbtV68" target="_blank">How to Iterate Over Tensor in TensorFlow | Iteration in TensorFlow | TensorFlow Beginner Tutorial</a>


</div>