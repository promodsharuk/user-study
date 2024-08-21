<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.sparse_softmax_cross_entropy_with_logits" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.sparse_softmax_cross_entropy_with_logits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Computes sparse softmax cross entropy between `logits` and `labels`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels, logits, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Measures the probability error in discrete classification tasks in which the
classes are mutually exclusive (each entry is in exactly one class).  For
example, each CIFAR-10 image is labeled with one and only one label: an image
can be a dog or a truck, but not both.

Note:  For this operation, the probability of a given label is considered
exclusive.  That is, soft classes are not allowed, and the `labels` vector
must provide a single specific index for the true class for each row of
`logits` (each minibatch entry).  For soft softmax classification with
a probability distribution for each entry, see
`softmax_cross_entropy_with_logits_v2`.

Warning: This op expects unscaled logits, since it performs a `softmax`
on `logits` internally for efficiency.  Do not call this op with the
output of `softmax`, as it will produce incorrect results.

A common use case is to have logits of shape
`[batch_size, num_classes]` and have labels of shape
`[batch_size]`, but higher dimensions are supported, in which
case the `dim`-th dimension is assumed to be of size `num_classes`.
`logits` must have the dtype of `float16`, `float32`, or `float64`, and
`labels` must have the dtype of `int32` or `int64`.

```
>>> logits = tf.constant([[2., -5., .5, -.1],
...                       [0., 0., 1.9, 1.4],
...                       [-100., 100., -100., -100.]])
>>> labels = tf.constant([0, 3, 1])
>>> tf.nn.sparse_softmax_cross_entropy_with_logits(
...     labels=labels, logits=logits).numpy()
array([0.29750752, 1.1448325 , 0.        ], dtype=float32)
```

To avoid confusion, passing only named arguments to this function is
recommended.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`<a id="labels"></a>
</td>
<td>
`Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
`labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
must be an index in `[0, num_classes)`. Other values will raise an
exception when this op is run on CPU, and return `NaN` for corresponding
loss and gradient rows on GPU.
</td>
</tr><tr>
<td>
`logits`<a id="logits"></a>
</td>
<td>
Unscaled log probabilities of shape `[d_0, d_1, ..., d_{r-1},
num_classes]` and dtype `float16`, `float32`, or `float64`.
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
A `Tensor` of the same shape as `labels` and of the same type as `logits`
with the softmax cross entropy loss.
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
If logits are scalars (need to have rank >= 1) or if the rank
of the labels is not equal to the rank of the logits minus one.
</td>
</tr>
</table>

