<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.layers.batch_normalization" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.layers.batch_normalization

<!-- Insert buttons and diff -->

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/legacy_tf_layers/normalization.py#L240-L474">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Functional interface for the batch normalization layer from_config(Ioffe et al., 2015).


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.layers.batch_normalization(
    inputs,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer=<a href="../../../../tf/compat/v1/zeros_initializer.md"><code>tf.compat.v1.zeros_initializer()</code></a>,
    gamma_initializer=<a href="../../../../tf/compat/v1/ones_initializer.md"><code>tf.compat.v1.ones_initializer()</code></a>,
    moving_mean_initializer=<a href="../../../../tf/compat/v1/zeros_initializer.md"><code>tf.compat.v1.zeros_initializer()</code></a>,
    moving_variance_initializer=<a href="../../../../tf/compat/v1/ones_initializer.md"><code>tf.compat.v1.ones_initializer()</code></a>,
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    training=False,
    trainable=True,
    name=None,
    reuse=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    virtual_batch_size=None,
    adjustment=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is a legacy api that is only compatible with eager execution and
<a href="../../../../tf/function.md"><code>tf.function</code></a> if you combine it with
`tf.compat.v1.keras.utils.track_tf1_style_variables`

Please refer to [tf.layers model mapping section of the migration guide]
(https://www.tensorflow.org/guide/migrate/model_mapping)
to learn how to use your TensorFlow v1 model in TF2 with Keras.

The corresponding TensorFlow v2 layer is
<a href="../../../../tf/keras/layers/BatchNormalization.md"><code>tf.keras.layers.BatchNormalization</code></a>.

The batch updating pattern with
<a href="../../../../tf/control_dependencies.md"><code>tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)</code></a> should not be used in
native TF2. Consult the <a href="../../../../tf/keras/layers/BatchNormalization.md"><code>tf.keras.layers.BatchNormalization</code></a> documentation
for further information.

#### Structural Mapping to Native TF2

None of the supported arguments have changed name.

Before:

```python
 x_norm = tf.compat.v1.layers.batch_normalization(x)
```

After:

To migrate code using TF1 functional layers use the [Keras Functional API]
(https://www.tensorflow.org/guide/keras/functional):

```python
 x = tf.keras.Input(shape=(28, 28, 1),)
 y = tf.keras.layers.BatchNormalization()(x)
 model = tf.keras.Model(x, y)
```
#### How to Map Arguments

TF1 Arg Name              | TF2 Arg Name              | Note
:------------------------ | :------------------------ | :---------------
`name`                    | `name`                    | Layer base class
`trainable`               | `trainable`               | Layer base class
`axis`                    | `axis`                    | -
`momentum`                | `momentum`                | -
`epsilon`                 | `epsilon`                 | -
`center`                  | `center`                  | -
`scale`                   | `scale`                   | -
`beta_initializer`        | `beta_initializer`        | -
`gamma_initializer`       | `gamma_initializer`       | -
`moving_mean_initializer` | `moving_mean_initializer` | -
`beta_regularizer`        | `beta_regularizer'        | -
`gamma_regularizer`       | `gamma_regularizer'       | -
`beta_constraint`         | `beta_constraint'         | -
`gamma_constraint`        | `gamma_constraint'        | -
`renorm`                  | Not supported             | -
`renorm_clipping`         | Not supported             | -
`renorm_momentum`         | Not supported             | -
`fused`                   | Not supported             | -
`virtual_batch_size`      | Not supported             | -
`adjustment`              | Not supported             | -


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Note: when training, the moving_mean and moving_variance need to be updated.
By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
need to be executed alongside the `train_op`. Also, be sure to add any
batch_normalization ops before getting the update_ops collection. Otherwise,
update_ops will be empty, and training/inference will not work properly. For
example:

```python
  x_norm = tf.compat.v1.layers.batch_normalization(x, training=training)

  # ...

  update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
  train_op = optimizer.minimize(loss)
  train_op = tf.group([train_op, update_ops])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
Tensor input.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
An `int`, the axis that should be normalized (typically the features
axis). For instance, after a `Convolution2D` layer with
`data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
</td>
</tr><tr>
<td>
`momentum`<a id="momentum"></a>
</td>
<td>
Momentum for the moving average.
</td>
</tr><tr>
<td>
`epsilon`<a id="epsilon"></a>
</td>
<td>
Small float added to variance to avoid dividing by zero.
</td>
</tr><tr>
<td>
`center`<a id="center"></a>
</td>
<td>
If True, add offset of `beta` to normalized tensor. If False,
`beta` is ignored.
</td>
</tr><tr>
<td>
`scale`<a id="scale"></a>
</td>
<td>
If True, multiply by `gamma`. If False, `gamma` is not used. When
the next layer is linear (also e.g. `nn.relu`), this can be disabled
since the scaling can be done by the next layer.
</td>
</tr><tr>
<td>
`beta_initializer`<a id="beta_initializer"></a>
</td>
<td>
Initializer for the beta weight.
</td>
</tr><tr>
<td>
`gamma_initializer`<a id="gamma_initializer"></a>
</td>
<td>
Initializer for the gamma weight.
</td>
</tr><tr>
<td>
`moving_mean_initializer`<a id="moving_mean_initializer"></a>
</td>
<td>
Initializer for the moving mean.
</td>
</tr><tr>
<td>
`moving_variance_initializer`<a id="moving_variance_initializer"></a>
</td>
<td>
Initializer for the moving variance.
</td>
</tr><tr>
<td>
`beta_regularizer`<a id="beta_regularizer"></a>
</td>
<td>
Optional regularizer for the beta weight.
</td>
</tr><tr>
<td>
`gamma_regularizer`<a id="gamma_regularizer"></a>
</td>
<td>
Optional regularizer for the gamma weight.
</td>
</tr><tr>
<td>
`beta_constraint`<a id="beta_constraint"></a>
</td>
<td>
An optional projection function to be applied to the
`beta` weight after being updated by an `Optimizer` (e.g. used to
implement norm constraints or value constraints for layer weights). The
function must take as input the unprojected variable and must return the
projected variable (which must have the same shape). Constraints are not
safe to use when doing asynchronous distributed training.
</td>
</tr><tr>
<td>
`gamma_constraint`<a id="gamma_constraint"></a>
</td>
<td>
An optional projection function to be applied to the
`gamma` weight after being updated by an `Optimizer`.
</td>
</tr><tr>
<td>
`training`<a id="training"></a>
</td>
<td>
Either a Python boolean, or a TensorFlow boolean scalar tensor
(e.g. a placeholder). Whether to return the output in training mode
(normalized with statistics of the current batch) or in inference mode
(normalized with moving statistics). **NOTE**: make sure to set this
  parameter correctly, or else your training/inference will not work
  properly.
</td>
</tr><tr>
<td>
`trainable`<a id="trainable"></a>
</td>
<td>
Boolean, if `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
String, the name of the layer.
</td>
</tr><tr>
<td>
`reuse`<a id="reuse"></a>
</td>
<td>
Boolean, whether to reuse the weights of a previous layer by the
same name.
</td>
</tr><tr>
<td>
`renorm`<a id="renorm"></a>
</td>
<td>
Whether to use Batch Renormalization (Ioffe, 2017). This adds
extra variables during training. The inference is the same for either
value of this parameter.
</td>
</tr><tr>
<td>
`renorm_clipping`<a id="renorm_clipping"></a>
</td>
<td>
A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
scalar `Tensors` used to clip the renorm correction. The correction `(r,
d)` is used as `corrected_value = normalized_value * r + d`, with `r`
clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
dmax are set to inf, 0, inf, respectively.
</td>
</tr><tr>
<td>
`renorm_momentum`<a id="renorm_momentum"></a>
</td>
<td>
Momentum used to update the moving means and standard
deviations with renorm. Unlike `momentum`, this affects training and
should be neither too small (which would add noise) nor too large (which
would give stale estimates). Note that `momentum` is still applied to
get the means and variances for inference.
</td>
</tr><tr>
<td>
`fused`<a id="fused"></a>
</td>
<td>
if `None` or `True`, use a faster, fused implementation if
possible.  If `False`, use the system recommended implementation.
</td>
</tr><tr>
<td>
`virtual_batch_size`<a id="virtual_batch_size"></a>
</td>
<td>
An `int`. By default, `virtual_batch_size` is `None`,
which means batch normalization is performed across the whole batch.
When `virtual_batch_size` is not `None`, instead perform "Ghost Batch
Normalization", which creates virtual sub-batches which are each
normalized separately (with shared gamma, beta, and moving statistics).
Must divide the actual batch size during execution.
</td>
</tr><tr>
<td>
`adjustment`<a id="adjustment"></a>
</td>
<td>
A function taking the `Tensor` containing the (dynamic) shape
of the input tensor and returning a pair (scale, bias) to apply to the
normalized values (before gamma and beta), only during training. For
example, if axis==-1,
  `adjustment = lambda shape: (
    tf.random.uniform(shape[-1:], 0.93, 1.07),
    tf.random.uniform(shape[-1:], -0.1, 0.1))` will scale the normalized
      value by up to 7% up or down, then shift the result by up to 0.1
      (with independent scaling and bias for each feature but shared
      across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Output tensor.
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
if eager execution is enabled.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">References</h2></th></tr>
<tr class="alt">
<td colspan="2">
Batch Normalization - Accelerating Deep Network Training by Reducing
Internal Covariate Shift:
  [Ioffe et al., 2015](http://proceedings.mlr.press/v37/ioffe15.html)
  ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
Batch Renormalization - Towards Reducing Minibatch Dependence in
Batch-Normalized Models:
  [Ioffe,
  2017](http://papers.nips.cc/paper/6790-batch-renormalization-towards-reducing-minibatch-dependence-in-batch-normalized-models)
  ([pdf](http://papers.nips.cc/paper/6790-batch-renormalization-towards-reducing-minibatch-dependence-in-batch-normalized-models.pdf))
</td>
</tr>

</table>


</div>

<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target"></span>
**Issue:** How does the `trainable` argument in `tf.compat.v1.layers.batch_normalization` affect the batch normalization layer in TensorFlow?

### Explanation
The `tf.compat.v1.layers.batch_normalization` function includes a `trainable` argument, which is a boolean. When set to `True`, this argument ensures that the variables associated with the batch normalization layer are added to the `GraphKeys.TRAINABLE_VARIABLES` collection. This collection is a subset of `GraphKeys.GLOBAL_VARIABLES` and includes variables that will be trained by an optimizer.

The primary trainable parameters in batch normalization are the scaling factor (gamma) and the offset (beta). These parameters are adjusted during training to normalize the input data. The moving averages, such as moving mean and moving variance, are not considered trainable parameters because they are updated based on the statistics of the current batch and are not directly optimized to minimize the loss function.

The `trainable` argument influences the behavior of the batch normalization layer by determining whether the gamma and beta variables are included in the `GraphKeys.TRAINABLE_VARIABLES` collection. If `trainable` is set to `True`, these variables will be optimized during training. If set to `False`, they will not be included in the trainable variables collection, and their values will remain fixed during training.

### Code Example
This code example demonstrates how the `trainable` parameter in `tf.compat.v1.layers.batch_normalization` influences the behavior of batch normalization.
```python
import tensorflow as tf
import numpy as np
# Define input tensor
inputs = tf.random.normal([10, 5, 5, 3])

# Define batch normalization layer with trainable=True
batch_norm_trainable = tf.compat.v1.layers.batch_normalization(
    inputs=inputs,
    trainable=True,
    training=True
)

# Define batch normalization layer with trainable=False
batch_norm_non_trainable = tf.compat.v1.layers.batch_normalization(
    inputs=inputs,
    trainable=False,
    training=True
)

# Initialize global variables
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    
    # Get trainable variables
    trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    
    # Get global variables
    global_vars = tf.compat.v1.global_variables()
    
    print('Trainable Variables:')
    for var in trainable_vars:
        print(var.name)
    
    print('\nGlobal Variables:')
    for var in global_vars:
        print(var.name)
```

### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/58097348/whats-the-differences-between-tf-graphkeys-global-variables-and-tf-graphkeys-tr" target="_blank">What's the differences between tf.GraphKeys.GLOBAL_VARIABLES and tf.GraphKeys.TRAINABLE_VARIABLES in tensorflow?</a>

- <a href="https://stackoverflow.com/questions/55619070/graphkeys-trainable-variables-vs-tf-trainable-variables" target="_blank">GraphKeys.TRAINABLE_VARIABLES vs tf.trainable_variables()</a>


</div>