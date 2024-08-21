<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.SensitivityAtSpecificity" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.SensitivityAtSpecificity

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">
<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/confusion_metrics.py#L771-L875">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>




Computes best sensitivity where specificity is >= specified value.

Inherits From: [`Metric`](../../../tf/keras/metrics/Metric.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.SensitivityAtSpecificity(
    specificity, num_thresholds=200, class_id=None, name=None, dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

the sensitivity at a given specificity.

`Sensitivity` measures the proportion of actual positives that are correctly
identified as such (tp / (tp + fn)).
`Specificity` measures the proportion of actual negatives that are correctly
identified as such (tn / (tn + fp)).

This metric creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the sensitivity at the given specificity. The threshold for the
given specificity value is computed and used to evaluate the corresponding
sensitivity.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold
predictions, and computing the fraction of them for which `class_id` is
indeed a correct label.

For additional information about specificity and sensitivity, see
[the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`specificity`<a id="specificity"></a>
</td>
<td>
A scalar value in range `[0, 1]`.
</td>
</tr><tr>
<td>
`num_thresholds`<a id="num_thresholds"></a>
</td>
<td>
(Optional) The number of thresholds to
use for matching the given specificity. Defaults to `200`.
</td>
</tr><tr>
<td>
`class_id`<a id="class_id"></a>
</td>
<td>
(Optional) Integer class ID for which we want binary metrics.
This must be in the half-open interval `[0, num_classes)`, where
`num_classes` is the last dimension of predictions.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
(Optional) string name of the metric instance.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
(Optional) data type of the metric result.
</td>
</tr>
</table>



#### Standalone usage:



```
>>> m = tf.keras.metrics.SensitivityAtSpecificity(0.5)
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
>>> m.result().numpy()
0.5
```

```
>>> m.reset_state()
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
...                sample_weight=[1, 1, 2, 2, 1])
>>> m.result().numpy()
0.333333
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.SensitivityAtSpecificity()])
```

## Methods

<h3 id="merge_state"><code>merge_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/base_metric.py#L288-L326">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_state(
    metrics
)
</code></pre>

Merges the state from one or more metrics.

This method can be used by distributed systems to merge the state
computed by different metric instances. Typically the state will be
stored in the form of the metric's weights. For example, a
tf.keras.metrics.Mean metric contains a list of two weight values: a
total and a count. If there were two instances of a
tf.keras.metrics.Accuracy that each independently aggregated partial
state for an overall accuracy calculation, these two metric's states
could be combined as follows:

```
>>> m1 = tf.keras.metrics.Accuracy()
>>> _ = m1.update_state([[1], [2]], [[0], [2]])
```

```
>>> m2 = tf.keras.metrics.Accuracy()
>>> _ = m2.update_state([[3], [4]], [[3], [4]])
```

```
>>> m2.merge_state([m1])
>>> m2.result().numpy()
0.75
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`metrics`
</td>
<td>
an iterable of metrics. The metrics must have compatible
state.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the provided iterable does not contain metrics matching
the metric's required specifications.
</td>
</tr>
</table>



<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/confusion_metrics.py#L727-L740">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets all of the metric state variables.

This function is called between epochs/steps,
when a metric is evaluated during training.

<h3 id="result"><code>result</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/confusion_metrics.py#L856-L867">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>result()
</code></pre>

Computes and returns the scalar metric value tensor or a dict of scalars.

Result computation is an idempotent operation that simply calculates the
metric value using the state variables.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A scalar tensor, or a dictionary of scalar tensors.
</td>
</tr>

</table>



<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/confusion_metrics.py#L699-L725">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates confusion matrix statistics.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`y_true`
</td>
<td>
The ground truth values.
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted values.
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional weighting of each example. Can
be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
and must be broadcastable to `y_true`. Defaults to `1`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Update op.
</td>
</tr>

</table>
</div>

<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target"></span>
**Issue:** How is the `num_thresholds` argument interpreted in `tf.keras.metrics.SensitivityAtSpecificity`?

### Explanation
The `num_thresholds` parameter determines the number of thresholds to be used for matching the given specificity. A threshold in this context is a value in the range [0,1] that determines whether a prediction is considered positive or negative. For example, with a threshold of 0.5, predictions greater than 0.5 are considered positive, and those less than or equal to 0.5 are considered negative.

When `num_thresholds` is set to 1, the only threshold considered is 0.5. This means that the metric will evaluate the specificity and sensitivity at this single threshold. If `num_thresholds` is greater than 1, the interval (0,1) is split into `num_thresholds - 1` equal sub-intervals, and these split points, along with 0 and 1, are used as additional thresholds. For instance, if `num_thresholds` is 2, the thresholds are [0, 1]; if `num_thresholds` is 3, the thresholds are [0, 0.5, 1]; and so on.

The metric computes the specificity at each threshold and selects the threshold that has the closest specificity to the specified value. It then computes the sensitivity at this selected threshold and returns it as the metric value. 

### Code Example
This example demonstrates how to interpret the `num_thresholds` argument in `tf.keras.metrics.SensitivityAtSpecificity`. When increasing `num_thresholds` to values greater than 1 (e.g., 2 or 200) results in a metric value of 1.0. This occurs because the data leads to ambiguous thresholds where any threshold in the range (0, 0.5) gives a specificity of 0.5, but the sensitivity value changes dramatically at 0.3.
```python
import tensorflow as tf
def print_metric_value(num_thresholds):
    # Create the SensitivityAtSpecificity metric with the given specificity and num_thresholds
    m = tf.keras.metrics.SensitivityAtSpecificity(0.4, num_thresholds=num_thresholds)
    # Update the state with true labels and predicted scores
    m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    # Print the result of the metric
    print('Result with num_thresholds = %d: %.1f' % (num_thresholds, m.result().numpy()))

# Test the metric with different num_thresholds values
print_metric_value(1)    # Expected output: 0.5
print_metric_value(2)    # Expected output: 1.0
print_metric_value(200)  # Expected output: 1.0
```
### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/56286350/tf-keras-metrics-specificityatsensitivity-num-thresholds-interpretation" target="_blank">tf.keras.metrics.SpecificityAtSensitivity num_thresholds interpretation</a>

</div>