import numpy as np
import plotly.graph_objects as go

def make_diag_stack_matrix(matrix_list):
    """
    行列のリストから対角方向に結合した行列を作成する
    """
    dim_i = sum([m.shape[0] for m in matrix_list])
    dim_j = sum([m.shape[1] for m in matrix_list])
    block_diag = np.zeros((dim_i, dim_j))

    pos_i = pos_j = 0
    for m in matrix_list:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                block_diag[pos_i+i, pos_j+j] = m[i, j]
        pos_i += m.shape[0]
        pos_j += m.shape[1]

    return block_diag

def make_hstack_matrix(matrix_list):
    """
    行列のリストから横方向に結合した行列を作成する
    """
    return np.concatenate(matrix_list, 1)

def plot(request):
    traces = request["traces"]
    config = request["config"]

    fig = go.Figure()
    for key in traces:
        fig.add_trace(
            go.Scatter(
                x=traces[key]["x"],
                y=traces[key]["y"],
                marker_color=traces[key]["color"],
                name=key,
                mode="lines",
                fill=traces[key]["fill"]
            )
        )

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_yaxes(range=[config["yaxis"]["min"], config["yaxis"]["max"]])
    fig.update_layout(showlegend=True)
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(count=5,
                        label="5y",
                        step="year",
                        stepmode="backward"),
                    dict(count=10,
                        label="10y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    return fig