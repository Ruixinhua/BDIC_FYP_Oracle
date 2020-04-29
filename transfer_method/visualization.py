from dimension_reduction import get_reduction_result
from model_output import get_model_output_df
from cluster import get_cluster_output_df
import os
import plotly.express as px
import math
import pickle
import configuration
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import tools


def plot_data(input_df, title, saved_path=None, **params):
    """
    Plot data using plotly and saved as html file
    Args:
        input_df: A pandas DataFrame that need to be plot, it must has columns "x", "y", "label", "type"
        title: The title of plot image
        saved_path: The path where the plot is stored
        **params: The parameters of when plot

    Returns: fig

    """
    if not saved_path:
        saved_path = os.path.join("plot", "%s.html" % title)
    if params is None:
        params = {"x": "x", "y": "y", "color": "label", "size_max": 20, "hover_data": ["type"]}
    fig = px.scatter(input_df, title=title, **params)
    # fig.update_traces(marker=dict(line=dict(width=0.1)), selector=dict(mode='markers'))
    fig.write_html(saved_path)
    print("Saved the plot in %s" % saved_path)
    return fig


def split_feature(input_df):
    """
    Split 2-D feature into x and y columns
    Args:
        input_df: A pandas DataFrame with column "feature"

    Returns: A pandas DataFrame with columns "x" and "y"

    """
    input_df["x"] = input_df.feature.apply(lambda p: p[0])
    input_df["y"] = input_df.feature.apply(lambda p: p[1])
    return input_df


def plot_by_dash(input_df, i, **plot_params):
    """
    Using dash to plot the DataFrame
    Args:
        input_df: A pandas DataFrame with columns "x", "y", "label", "type" or "center
        i: The index number of this plot
        **plot_params: some plot parameters used to define the plot

    Returns:

    """
    dimensions = ["x", "y", "color", "facet_col", "facet_row", "size"]
    col_options = [dict(label=x, value=x) for x in input_df.columns]

    div_option = [html.Div([
        html.P(["Task:", dcc.Dropdown(id="%s_%s" % ("task", i), options=[
            dict(label=x, value=x) for x in ["cluster", "debug"]], value="cluster")])
    ])]
    div_option.extend([
        html.P([d + ":", dcc.Dropdown(id="%s_%s" % (d, i), options=col_options)])
        for d in dimensions
    ])
    div = html.Div(
        [
            html.Div(div_option, style={"width": "25%", "float": "left"}),
            dcc.Graph(id="graph_%s" % i, style={"width": "75%", "display": "inline-block"}),
        ]
    )

    input_val = [Input("%s_%s" % (d, i), "value") for d in dimensions]
    input_val.append(Input("%s_%s" % ("task", i), "value"))

    @app.callback(Output("graph_%s" % i, "figure"), input_val)
    def make_figure(x, y, color, facet_col, facet_row, size, task):
        if task == "cluster":
            x, y = "x", "y"
            color = "label" if color is not None else color
            facet_col = "type" if facet_col is not None else facet_col
            facet_row = "center" if "center" in input_df.columns and facet_row is not None else None
        if plot_params is not None and not x and not y and not color and not facet_col and not facet_row and not size:
            return px.scatter(input_df, **plot_params)
        return px.scatter(input_df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, size=size,
                          size_max=20, hover_data=["file_name", "size"], height=700)

    return div


if __name__ == "__main__":
    # test code
    new_dir = os.path.join(configuration.dataset_root_dir, "paired_jia_jin_sep-%s" % configuration.set_iter_no)
    outlier_dir = os.path.join(configuration.dataset_root_dir, "paired_jia_jin_out-%s" % configuration.set_iter_no)
    model_type = "vae"
    test_num = 20
    # model_paths = ("model/jia_%s_base_full.pkl" % model_type, "model/jin_%s_base_full.pkl" % model_type)
    model_paths = ("model/jia_%s_base_no_kld.pkl" % model_type, "model/jin_%s_base_no_kld.pkl" % model_type)
    # model_paths = ("model/jia_%s_base_full_old.pkl" % model_type, "model/jin_%s_base_full_old.pkl" % model_type)
    models = [tools.get_model_by_state(path, tools.get_default_model_class(model_type)) for path in model_paths]
    configuration.create_dirs(["output/", "plot/", "reduction/", "cluster/"])
    output_file = os.path.join("output", "%s_nk_%d-%d.pkl" % (model_type, test_num, configuration.set_iter_no))
    reduction_file = os.path.join("reduction", "%s_nk-%d-%d.pkl" % (model_type, test_num, configuration.set_iter_no))
    cluster_file = os.path.join("cluster", "%s_nk-%d-%d.pkl" % (model_type, test_num, configuration.set_iter_no))
    char_list = pickle.load(open("char_list.pkl", "rb"))
    divs = []
    app = dash.Dash(
        __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    )
    for index in range(math.ceil(len(char_list) / test_num)):
        if index == 2: break
        begin, end = test_num * index, test_num * (index + 1)
        # get outputs from model
        model_outputs = get_model_output_df(models=models, model_type=model_type, output_file=output_file,
                                            begin=begin, end=end, debug=True)
        # get reduction results here which is a DataFrame{"label":[],"x":[],"y":[],"size":[],"path":[],"type":[]}
        model_reduction_results = get_reduction_result(model_outputs, output_file=reduction_file, debug=True)
        # get cluster result here
        model_cluster_result = get_cluster_output_df(model_reduction_results, output_file=cluster_file, debug=True)
        model_cluster_result = split_feature(model_cluster_result)
        print("After filter:", model_cluster_result.shape)
        test_chars = char_list[begin:end]
        divs.append(html.Div([html.H2("Character:" + ",".join(test_chars)), plot_by_dash(model_cluster_result, index)]))
    app.layout = html.Div(divs)
    app.run_server(debug=False, port=8050)
    # test plot
    # plot_params = {"x": "x", "y": "y", "color": "label", "size_max": 20, "hover_data": ["type"]}
    # plot_data(model_cluster_result, "jia-jin-reduction-%s-%d" % (model_type, test_num), **plot_params)
    # plot_data(model_cluster_result, "jia-jin-centers-%s-%d" % (model_type, test_num), **plot_params)
