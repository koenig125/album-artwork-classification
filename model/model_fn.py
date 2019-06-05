"""Define the model."""

import tensorflow as tf


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']

    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    dropout_rate = params.dropout_rate
    regularizer = tf.contrib.layers.l2_regularizer(scale=params.regularization_rate)
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8, num_channels * 8]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same', kernel_regularizer=regularizer)
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            if params.use_dropout:
                out = tf.layers.dropout(out, rate=dropout_rate, training=is_training)
            out = tf.layers.max_pooling2d(out, 2, 2)

    np = int(params.image_size / (2 ** len(channels))) # num "pixels" wide
    assert out.get_shape().as_list() == [None, np, np, num_channels * 8]

    out = tf.reshape(out, [-1, np * np * num_channels * 8])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, num_channels * 8, kernel_regularizer=regularizer)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        if params.use_dropout:
            out = tf.layers.dropout(out, rate=dropout_rate, training=is_training)
    with tf.variable_scope('fc_2'):
        out = tf.layers.dense(out, num_channels * 8, kernel_regularizer=regularizer)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        if params.use_dropout:
            out = tf.layers.dropout(out, rate=dropout_rate, training=is_training)
    with tf.variable_scope('fc_3'):
        logits = tf.layers.dense(out, params.num_labels, kernel_regularizer=regularizer)

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        # predictions = tf.cast(tf.one_hot(tf.argmax(logits, 1), params.num_labels), tf.int64)
        predictions = tf.argmax(logits, 1)

    # Define loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits) + tf.losses.get_regularization_loss()
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=predictions),
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # Add incorrectly labeled images
    mask = tf.not_equal(tf.argmax(labels, 1), predictions)
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
