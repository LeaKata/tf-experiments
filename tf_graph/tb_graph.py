from __future__ import annotations

import datetime
import tensorflow as tf

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_log_dir = "logs/gradient_tape/" + current_time + "/test"
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


@tf.function
def my_multiply(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """soon"""
    return x * y


x = tf.constant(5)
y = tf.constant(10)

tf.summary.trace_on(graph=True, profiler=True)
z = my_multiply(x, y)
with test_summary_writer.as_default():
    tf.summary.trace_export(
        name="my_multiply",
        step=0,
        profiler_outdir=test_log_dir,
    )
