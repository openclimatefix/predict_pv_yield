from predict_pv_yield.visualisation.line import plot_batch_results, plot_one_result, make_trace
import numpy as np


def test_make_trace():

    x = np.random.random(7)
    y = np.random.random(7)

    _ = make_trace(x=x, y=y, truth=True)
    _ = make_trace(x=x, y=y, truth=False)
    _ = make_trace(x=x, y=y, truth=True, show_legend=False)


def test_plot_batch_results():

    size = (2,7)

    x = np.random.random(size)
    y = np.random.random(size)
    y_hat = np.random.random(size)

    _ = plot_batch_results(x=x, y=y, y_hat=y_hat, model_name='test_model')


def test_plot_one_result():

    x = np.random.random(7)
    y = np.random.random(7)
    y_hat = np.random.random(7)

    _ = plot_one_result(x=x, y=y, y_hat=y_hat)
