<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.feature_column.categorical_column_with_hash_bucket" />
<meta itemprop="path" content="Stable" />
</div>

# tf.feature_column.categorical_column_with_hash_bucket

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/feature_column/feature_column_v2.py">View source</a>

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">

Represents sparse feature where ids are set by hashing. (deprecated)



Warning: tf.feature_column is not recommended for new code. Instead,
feature preprocessing can be done directly using either [Keras preprocessing
layers](https://www.tensorflow.org/guide/migrate/migrating_feature_columns)
or through the one-stop utility [`tf.keras.utils.FeatureSpace`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/FeatureSpace)
built on top of them. See the [migration guide](https://tensorflow.org/guide/migrate)
for details.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.feature_column.categorical_column_with_hash_bucket`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.feature_column.categorical_column_with_hash_bucket(
    key,
    hash_bucket_size,
    dtype=<a href="../../tf/dtypes.md#string"><code>tf.dtypes.string</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use Keras preprocessing layers instead, either directly or via the <a href="../../tf/keras/utils/FeatureSpace.md"><code>tf.keras.utils.FeatureSpace</code></a> utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.

Use this when your sparse features are in string or integer format, and you
want to distribute your inputs into a finite number of buckets by hashing.
output_id = Hash(input_feature_string) % bucket_size for string type input.
For int type input, the value is converted to its string representation first
and then hashed by the same formula.

For input dictionary `features`, `features[key]` is either `Tensor` or
`SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
and `''` for string, which will be dropped by this feature column.

#### Example:



```python
import tensorflow as tf
keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
10000)
columns = [keywords]
features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
'LSTM', 'Keras', 'RNN']])}
linear_prediction, _, _ = tf.compat.v1.feature_column.linear_model(features,
columns)

# or
import tensorflow as tf
keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
10000)
keywords_embedded = tf.feature_column.embedding_column(keywords, 16)
columns = [keywords_embedded]
features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
'LSTM', 'Keras', 'RNN']])}
input_layer = tf.keras.layers.DenseFeatures(columns)
dense_tensor = input_layer(features)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`key`<a id="key"></a>
</td>
<td>
A unique string identifying the input feature. It is used as the column
name and the dictionary key for feature parsing configs, feature `Tensor`
objects, and feature columns.
</td>
</tr><tr>
<td>
`hash_bucket_size`<a id="hash_bucket_size"></a>
</td>
<td>
An int > 1. The number of buckets.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The type of features. Only string and integer types are supported.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `HashedCategoricalColumn`.
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
`hash_bucket_size` is not greater than 1.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
`dtype` is neither string nor integer.
</td>
</tr>
</table>

</div>

<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target"></span>
**Issue:** How to use `tf.feature_column.categorical_column_with_hash_bucket` to apply feature columns directly to a `tf.data.Dataset` without using a `tf.Estimator`?

### Explanation
To apply feature columns directly to a `tf.data.Dataset` without using a `tf.Estimator`, the `tf.feature_column.categorical_column_with_hash_bucket` API offers an efficient solution for handling categorical features. This method hashes feature values into a fixed number of buckets, making it scalable for large vocabularies or unknown categories. After defining your categorical columns using this API, you can convert the hashed indices into dense or one-hot encoded representations with `tf.feature_column.indicator_column`, allowing easy integration into the dataset pipeline.

The feature columns are then applied using `tf.keras.layers.DenseFeatures` within a `tf.data.Dataset` pipeline. This approach enables you to preprocess and encode features without needing an Estimator, allowing for seamless integration into TensorFlow's eager execution environment.

### Code Example
This example demonstrates how to use `tf.feature_column.categorical_column_with_hash_bucket` to apply feature columns directly to a tf.data.Dataset without using tf.Estimator.
```python

import tensorflow as tf
# Define column names
column_names = ['UserID', 'MovieID']

# Create feature columns
user_col = tf.feature_column.categorical_column_with_hash_bucket(key='UserID', hash_bucket_size=1000)
movie_col = tf.feature_column.categorical_column_with_hash_bucket(key='MovieID', hash_bucket_size=1000)
feature_columns = [tf.feature_column.indicator_column(user_col), tf.feature_column.indicator_column(movie_col)]

# Create a DenseFeatures layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns=feature_columns)

# Function to process each line of the CSV
def process_csv(line):
    fields = tf.io.decode_csv(line, record_defaults=[tf.constant([], dtype=tf.int32)]*2, field_delim=';')
    features = dict(zip(column_names, fields))
    return features

# Create a TextLineDataset
csv_filepath = 'path/to/your/csvfile.csv'  # Update with your CSV file path
ds = tf.data.TextLineDataset(csv_filepath)

# Map the process_csv function to each element in the dataset
ds = ds.map(process_csv, num_parallel_calls=4)

# Batch the dataset
ds = ds.batch(10)

# Apply the feature layer to the dataset
ds = ds.map(lambda x: feature_layer(x))

# Iterate through the dataset and print the results
for batch in ds:
    print(batch)
```
### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/57403472/how-do-i-add-a-new-feature-column-to-a-tf-data-dataset-object" target="_blank">How do I add a new feature column to a tf.data.Dataset object?</a>


</div>


