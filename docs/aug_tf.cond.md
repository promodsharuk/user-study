<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.cond" />
<meta itemprop="path" content="Stable" />
</div>

# tf.cond

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/cond.py">View source</a>

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">

Return `true_fn()` if the predicate `pred` is true else `false_fn()`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.cond(
    pred, true_fn=None, false_fn=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note: This op is automatically used in a <a href="../tf/function.md"><code>tf.function</code></a> to convert Python
if-statements when the predicate is a <a href="../tf/Tensor.md"><code>tf.Tensor</code></a>, unless `autograph=False` is
explicitly specified in <a href="../tf/function.md"><code>tf.function</code></a> args. For example, the following are
equivalent:

```
>>> @tf.function
... def fun1(x,y):
...   if x > 0:  # AutoGraph converts if-statement to tf.cond().
...     z = y+1
...   else:
...     z = y-1
...   return z
>>> fun1(tf.constant(7), tf.constant(3)).numpy()
4
```

```
>>> @tf.function
... def fun2(x,y):
...   pred = x > 0
...   true_fn =  lambda: y+1
...   false_fn = lambda: y-1
...   return tf.cond(pred, true_fn, false_fn)  # Use tf.cond() explicitly.
>>> fun1(tf.constant(7), tf.constant(3)).numpy()
4
```

For more information, see [tf.function and AutoGraph guide](
https://www.tensorflow.org/guide/function#autograph_transformations).

`true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
`false_fn` must have the same non-zero number and type of outputs.

**WARNING**: Any Tensors or Operations created outside of `true_fn` and
`false_fn` will be executed regardless of which branch is selected at runtime.

Although this behavior is consistent with the dataflow model of TensorFlow,
it has frequently surprised users who expected a lazier semantics.
Consider the following simple program:

```
>>> x, y = tf.constant(2, dtype=tf.int32), tf.constant(4, dtype=tf.int32)
>>> z = tf.multiply(x, y)
>>> r = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
>>> r.numpy()
10
```

If `x < y`, the `tf.add` operation will be executed and `tf.square`
operation will not be executed. Since `z` is needed for at least one
branch of the `cond`, the <a href="../tf/math/multiply.md"><code>tf.multiply</code></a> operation is always executed,
unconditionally.

Note that `cond` calls `true_fn` and `false_fn` *exactly once* (inside the
call to `cond`, and not at all during `Session.run()`). `cond`
stitches together the graph fragments created during the `true_fn` and
`false_fn` calls with some additional graph nodes to ensure that the right
branch gets executed depending on the value of `pred`.

<a href="../tf/cond.md"><code>tf.cond</code></a> supports nested structures as implemented in
`tensorflow.python.util.nest`. Both `true_fn` and `false_fn` must return the
same (possibly nested) value structure of lists, tuples, and/or named tuples.
Singleton lists and tuples form the only exceptions to this: when returned by
`true_fn` and/or `false_fn`, they are implicitly unpacked to single values.

Note: It is illegal to "directly" use tensors created inside a cond branch
outside it, e.g. by storing a reference to a branch tensor in the python
state. If you need to use a tensor created in a branch function you should
return it as an output of the branch function and use the output from
<a href="../tf/cond.md"><code>tf.cond</code></a> instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`pred`<a id="pred"></a>
</td>
<td>
A scalar determining whether to return the result of `true_fn` or
`false_fn`.
</td>
</tr><tr>
<td>
`true_fn`<a id="true_fn"></a>
</td>
<td>
The callable to be performed if pred is true.
</td>
</tr><tr>
<td>
`false_fn`<a id="false_fn"></a>
</td>
<td>
The callable to be performed if pred is false.
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
Tensors returned by the call to either `true_fn` or `false_fn`. If the
callables return a singleton list, the element is extracted from the list.
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
if `true_fn` or `false_fn` is not callable.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `true_fn` and `false_fn` do not return the same number of
tensors, or return tensors of different types.
</td>
</tr>
</table>



#### Example:



```
>>> x = tf.constant(2)
>>> y = tf.constant(5)
>>> def f1(): return tf.multiply(x, 7)
>>> def f2(): return tf.add(y, 3)
>>> r = tf.cond(tf.less(x, y), f1, f2)
>>> # r is set to f1().
>>> # Operations in f2 (e.g., tf.add) are not executed.
>>> r.numpy()
14
```
</div>


<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">

# You might need to know !
<span id="target1"></span>
**Issue:** Why does `tf.cond` produce unexpected results when `sess.run` is called multiple times, causing `deterministic_action` to change unexpectedly?

### Explanation 
In TensorFlow, the `tf.cond` function is used to return the result of `true_fn` if the predicate `pred` is true, otherwise it returns the result of `false_fn`. However, it is important to note that any Tensors or Operations created outside of `true_fn` and `false_fn` will be executed regardless of which branch is selected at runtime. This behavior is consistent with TensorFlow's dataflow model but can be surprising to users expecting lazier semantics.

In the provided example, `deterministic_action` and `random_action` are both generated using `tf.random_uniform` with fixed seeds. The `chose_random` tensor is determined to be `False` because `eps` is initialized to `0`. Therefore, `stochastic_action` should be equal to `deterministic_action`.

The unexpected output (`s_action=1` instead of `4`) occurs because the random number generation is evaluated again in a new `sess.run` call. Each call to `sess.run` generates a new random number, which is why `deterministic_action` changes from `4` to `1` in subsequent evaluations. This behavior is due to the stateful nature of random number generation in TensorFlow, where the sequence of random numbers is maintained across session runs.

### Code Example
Below is a complete executable code example that addresses this issue.
```python

import tensorflow as tf
import numpy as np
with tf.Graph().as_default():
    with tf.device('/cpu:0'):
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        with tf.variable_scope('test_cond') as sc:
            deterministic_action = tf.random_uniform([], minval=0, maxval=15, dtype=tf.int64, seed=0) # 4
            random_action = tf.random_uniform([], minval=0, maxval=15, dtype=tf.int64, seed=1) # 11
            chose_random = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32) < eps # False because eps = 0
            stochastic_action = tf.cond(chose_random, lambda: random_action, lambda: deterministic_action) 

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init, feed_dict={stochastic_ph: True})
        d_action, r_action, e, c_action, s_action = sess.run([deterministic_action, random_action, eps, chose_random, stochastic_action])
        print ("det_action= ", d_action)
        print ("rand_action= ", r_action)
        print ("chose_rand= ", c_action)
        print ("s_action= ", s_action)
```
### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target1) related to this API.
- <a href="https://stackoverflow.com/questions/53079436/tensorflow-tf-cond-giving-unexpected-output" target="_blank">tensorflow Tf.cond giving unexpected output</a>

- <a href="https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond" target="_blank">Confused by the behavior of `tf.cond`</a>

- <a href="https://stackoverflow.com/questions/68444180/tf-case-and-tf-cond-executes-all-the-functions-within-in-tensorflow" target="_blank">`tf.case` and `tf.cond` executes all the functions within in TensorFlow</a>

### Related YouTube Tutorials
Video tutorials to learn concepts related to the [**issue**](#target1) with this API.
- <a href="https://www.youtube.com/watch?v=IzKXEbpT9Lg" target="_blank">Inside TensorFlow: Control Flow</a>

- <a href="https://www.youtube.com/watch?v=A4WJiS6sltw" target="_blank">Retval[0] does not have value : tf.cond(condition, net1, net2)</a>

</div>




<div style="padding: 5px; float: right; width: 34%; margin-left: 5px; height: 10px;"><br></div>




<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">

# You might need to know !
<span id="target2"></span>
**Issue:** 
Why does `tf.cond` not reflect changes to a boolean variable made after graph construction in TensorFlow?

### Explanation 
In TensorFlow, `tf.cond` is used to create conditional branches in the computation graph. The predicate `pred` is evaluated, and based on its value, either `true_fn` or `false_fn` is executed. However, it is important to understand that `tf.cond` evaluates the predicate and constructs the graph branches at graph construction time, not at runtime. This means that any changes to the boolean variable after the graph has been constructed will not affect the branching decision.

In the provided example, the boolean variable `foo` is converted to a tensor and passed to `tf.cond`. Since `foo` is evaluated at graph construction time, the initial value of `foo` determines the branch that will be executed, and subsequent changes to `foo` are not tracked.

To achieve dynamic branching based on updates to a boolean variable, you need to ensure that the predicate is a tensor that can be fed with different values at runtime. This can be done by using a placeholder or a variable that can be updated within the session. By feeding different values to the placeholder or updating the variable, you can control the branching decision dynamically.

### Code Example
The solution involves using TensorFlow's AutoGraph feature with the `@tf.function decorator`, which allows for dynamic control flow within the graph.
```python
import tensorflow as tf
@tf.function
def funa():
    return tf.constant(32)

@tf.function
def funb():
    return tf.constant(25)

@tf.function
def dynamic_cond(foo):
    return tf.cond(foo, funa, funb)

# Initialize the boolean variable
foo = tf.Variable(True)

# Create a list to store the results
results = []

# Run the dynamic condition in a loop
for i in range(20):
    if i > 10:
        foo.assign(False)
    results.append(dynamic_cond(foo).numpy())

print(results)

```
### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target2) related to this API.
- <a href="https://stackoverflow.com/questions/35833011/how-to-add-if-condition-in-a-tensorflow-graph" target="_blank">How to add if condition in a TensorFlow graph?</a>

- <a href="https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond" target="_blank">Confused by the behavior of `tf.cond`</a>


### Related YouTube Tutorials
Video tutorials to learn concepts related to the [**issue**](#target2) with this API.
- <a href="https://www.youtube.com/watch?v=IzKXEbpT9Lg" target="_blank">Inside TensorFlow: Control Flow</a>

- <a href="https://www.youtube.com/watch?v=jh4ITuOytE4" target="_blank">AutoGraph: Easy control flow for graphs (TensorFlow Tip of the Week)</a>
</div>