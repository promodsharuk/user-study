<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.make_csv_dataset" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.make_csv_dataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/ops/readers.py">View source</a>

<div style="border: 0px solid #ccc; padding: 5px; float: left; width: 63%;">

Reads CSV files into a dataset.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.make_csv_dataset(
    file_pattern,
    batch_size,
    column_names=None,
    column_defaults=None,
    label_name=None,
    select_columns=None,
    field_delim=&#x27;,&#x27;,
    use_quote_delim=True,
    na_value=&#x27;&#x27;,
    header=True,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=None,
    num_parallel_reads=None,
    sloppy=False,
    num_rows_for_inference=100,
    compression_type=None,
    ignore_errors=False,
    encoding=&#x27;utf-8&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Reads CSV files into a dataset, where each element of the dataset is a
(features, labels) tuple that corresponds to a batch of CSV rows. The features
dictionary maps feature column names to `Tensor`s containing the corresponding
feature data, and labels is a `Tensor` containing the batch's label data.

By default, the first rows of the CSV files are expected to be headers listing
the column names. If the first rows are not headers, set `header=False` and
provide the column names with the `column_names` argument.

By default, the dataset is repeated indefinitely, reshuffling the order each
time. This behavior can be modified by setting the `num_epochs` and `shuffle`
arguments.

For example, suppose you have a CSV file containing

| Feature_A | Feature_B |
| --------- | --------- |
| 1         | "a"       |
| 2         | "b"       |
| 3         | "c"       |
| 4         | "d"       |

```
# No label column specified
dataset = tf.data.experimental.make_csv_dataset(filename, batch_size=2)
iterator = dataset.as_numpy_iterator()
print(dict(next(iterator)))
# prints a dictionary of batched features:
# OrderedDict([('Feature_A', array([1, 4], dtype=int32)),
#              ('Feature_B', array([b'a', b'd'], dtype=object))])
```

```
# Set Feature_B as label column
dataset = tf.data.experimental.make_csv_dataset(
    filename, batch_size=2, label_name="Feature_B")
iterator = dataset.as_numpy_iterator()
print(next(iterator))
# prints (features, labels) tuple:
# (OrderedDict([('Feature_A', array([1, 2], dtype=int32))]),
#  array([b'a', b'b'], dtype=object))
```

See the
[Load CSV data guide](https://www.tensorflow.org/tutorials/load_data/csv) for
more examples of using `make_csv_dataset` to read CSV data.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`file_pattern`<a id="file_pattern"></a>
</td>
<td>
List of files or patterns of file paths containing CSV
records. See <a href="../../../tf/io/gfile/glob.md"><code>tf.io.gfile.glob</code></a> for pattern rules.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
An int representing the number of records to combine
in a single batch.
</td>
</tr><tr>
<td>
`column_names`<a id="column_names"></a>
</td>
<td>
An optional list of strings that corresponds to the CSV
columns, in order. One per column of the input record. If this is not
provided, infers the column names from the first row of the records.
These names will be the keys of the features dict of each dataset element.
</td>
</tr><tr>
<td>
`column_defaults`<a id="column_defaults"></a>
</td>
<td>
A optional list of default values for the CSV fields. One
item per selected column of the input record. Each item in the list is
either a valid CSV dtype (float32, float64, int32, int64, or string), or a
`Tensor` with one of the aforementioned types. The tensor can either be
a scalar default value (if the column is optional), or an empty tensor (if
the column is required). If a dtype is provided instead of a tensor, the
column is also treated as required. If this list is not provided, tries
to infer types based on reading the first num_rows_for_inference rows of
files specified, and assumes all columns are optional, defaulting to `0`
for numeric values and `""` for string values. If both this and
`select_columns` are specified, these must have the same lengths, and
`column_defaults` is assumed to be sorted in order of increasing column
index.
</td>
</tr><tr>
<td>
`label_name`<a id="label_name"></a>
</td>
<td>
A optional string corresponding to the label column. If
provided, the data for this column is returned as a separate `Tensor` from
the features dictionary, so that the dataset complies with the format
expected by a `tf.Estimator.train` or `tf.Estimator.evaluate` input
function.
</td>
</tr><tr>
<td>
`select_columns`<a id="select_columns"></a>
</td>
<td>
An optional list of integer indices or string column
names, that specifies a subset of columns of CSV data to select. If
column names are provided, these must correspond to names provided in
`column_names` or inferred from the file header lines. When this argument
is specified, only a subset of CSV columns will be parsed and returned,
corresponding to the columns specified. Using this results in faster
parsing and lower memory usage. If both this and `column_defaults` are
specified, these must have the same lengths, and `column_defaults` is
assumed to be sorted in order of increasing column index.
</td>
</tr><tr>
<td>
`field_delim`<a id="field_delim"></a>
</td>
<td>
An optional `string`. Defaults to `","`. Char delimiter to
separate fields in a record.
</td>
</tr><tr>
<td>
`use_quote_delim`<a id="use_quote_delim"></a>
</td>
<td>
An optional bool. Defaults to `True`. If false, treats
double quotation marks as regular characters inside of the string fields.
</td>
</tr><tr>
<td>
`na_value`<a id="na_value"></a>
</td>
<td>
Additional string to recognize as NA/NaN.
</td>
</tr><tr>
<td>
`header`<a id="header"></a>
</td>
<td>
A bool that indicates whether the first rows of provided CSV files
correspond to header lines with column names, and should not be included
in the data.
</td>
</tr><tr>
<td>
`num_epochs`<a id="num_epochs"></a>
</td>
<td>
An int specifying the number of times this dataset is repeated.
If None, cycles through the dataset forever.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
A bool that indicates whether the input should be shuffled.
</td>
</tr><tr>
<td>
`shuffle_buffer_size`<a id="shuffle_buffer_size"></a>
</td>
<td>
Buffer size to use for shuffling. A large buffer size
ensures better shuffling, but increases memory usage and startup time.
</td>
</tr><tr>
<td>
`shuffle_seed`<a id="shuffle_seed"></a>
</td>
<td>
Randomization seed to use for shuffling.
</td>
</tr><tr>
<td>
`prefetch_buffer_size`<a id="prefetch_buffer_size"></a>
</td>
<td>
An int specifying the number of feature
batches to prefetch for performance improvement. Recommended value is the
number of batches consumed per training step. Defaults to auto-tune.
</td>
</tr><tr>
<td>
`num_parallel_reads`<a id="num_parallel_reads"></a>
</td>
<td>
Number of threads used to read CSV records from files.
If >1, the results will be interleaved. Defaults to `1`.
</td>
</tr><tr>
<td>
`sloppy`<a id="sloppy"></a>
</td>
<td>
If `True`, reading performance will be improved at
the cost of non-deterministic ordering. If `False`, the order of elements
produced is deterministic prior to shuffling (elements are still
randomized if `shuffle=True`. Note that if the seed is set, then order
of elements after shuffling is deterministic). Defaults to `False`.
</td>
</tr><tr>
<td>
`num_rows_for_inference`<a id="num_rows_for_inference"></a>
</td>
<td>
Number of rows of a file to use for type inference
if record_defaults is not provided. If None, reads all the rows of all
the files. Defaults to 100.
</td>
</tr><tr>
<td>
`compression_type`<a id="compression_type"></a>
</td>
<td>
(Optional.) A <a href="../../../tf.md#string"><code>tf.string</code></a> scalar evaluating to one of
`""` (no compression), `"ZLIB"`, or `"GZIP"`. Defaults to no compression.
</td>
</tr><tr>
<td>
`ignore_errors`<a id="ignore_errors"></a>
</td>
<td>
(Optional.) If `True`, ignores errors with CSV file parsing,
such as malformed data or empty lines, and moves on to the next valid
CSV record. Otherwise, the dataset raises an error and stops processing
when encountering any invalid records. Defaults to `False`.
</td>
</tr><tr>
<td>
`encoding`<a id="encoding"></a>
</td>
<td>
Encoding to use when reading. Defaults to `UTF-8`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dataset, where each element is a (features, labels) tuple that corresponds
to a batch of `batch_size` CSV rows. The features dictionary maps feature
column names to `Tensor`s containing the corresponding column data, and
labels is a `Tensor` containing the column data for the label column
specified by `label_name`.
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
If any of the arguments is malformed.
</td>
</tr>
</table>
</div>


<div style="border: 1px solid #ccc; padding: 5px; float: right; width: 34%; margin-left: 5px; background-color: #e6ffe6;">
  
# You might need to know !
<span id="target1"></span>
**Issue:**  How to use the `map` transformation to add 2 to the 'weight' feature and remove the 'color' feature in a dataset created with `tf.data.experimental.make_csv_dataset`?

### Explanation
To modify a single feature in a TensorFlow dataset, such as adding 2 to the 'weight' feature, you can use the `map` transformation. The `map` function applies a transformation to each element of the dataset. In this case, you can define a lambda function that modifies the 'weight' feature by adding 2 to its value. The `map` function will then apply this transformation to each element in the dataset, resulting in the desired modification.

To remove a single feature from a TensorFlow dataset, you can use the `map` transformation to filter out the unwanted feature. The `map` function allows you to return a modified version of each element, excluding the feature you want to remove. By defining a lambda function that constructs a new dictionary without the specified feature, you can effectively drop the column from the dataset.

### Code Example
This example demonstrates how to modify a single feature in a TensorFlow dataset and how to remove a single feature from the dataset.

```python
import tensorflow as tf
import pandas as pd

# Create a sample CSV file
data = {'color': ['red', 'blue', 'green'], 'price': [120, 80, 90], 'weight': [1.2, 2.0, 3.0]}
df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)

# Load the dataset from the CSV file
dataset = tf.data.experimental.make_csv_dataset('data.csv', batch_size=1, num_epochs=1, shuffle=False)
# Function to modify the 'weight' feature
modify_weight = lambda x: {'color': x['color'], 'price': x['price'], 'weight': x['weight'] + 2}

# Apply the function to the dataset
dataset = dataset.map(modify_weight)

# Function to remove the 'color' feature
remove_color = lambda x: {'price': x['price'], 'weight': x['weight']}

dataset = dataset.map(remove_color)

# Print the modified dataset
for data in dataset:
    print(data)

```
### Related Stack Overflow Posts
Additional information to obtain knowledge needed to address the [**issue**](#target) related to this API.
- <a href="https://stackoverflow.com/questions/71149271/how-to-remove-single-feature-from-tensorflow-dataset-how-to-use-apply-on-single" target="_blank">How to remove single feature from tensorflow dataset, how to use apply on single feature?</a>

</div>