import plotly.express as px
import numpy as np

def prepare_CCM_plotly_heatmap_figs(ccm, ccm_bin, show_values, fmt, show):
    num_rows, num_cols = ccm.shape
    
    fig1 = px.imshow(ccm,
                     text_auto=fmt,
                     aspect="auto",
                     range_color=[5, np.max(ccm.values)],
                     color_continuous_scale="Reds",
                     labels=dict(x="", y="", color=""),
                     title='Co-occurrence Matrix Heatmap')
    
    fig1.update_layout(
        autosize=True,
        title_x=0.5,
        title_font=dict(size=30),
    )

    fig1.update_xaxes(
        tickfont=dict(size=20),
        tickangle=-90,
        ticktext=ccm.columns.tolist(),
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        tickcolor="black",
        showgrid=False
    )

    fig1.update_yaxes(
        tickfont=dict(size=20),
        ticktext=ccm.columns.tolist(),
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        tickcolor="black",
        showgrid=False
    )

    fig1.update_xaxes(constrain='domain')
    fig1.update_yaxes(scaleanchor="x", scaleratio=1)

    fig2 = px.imshow(ccm_bin,
                     text_auto=fmt,
                     aspect="auto",
                     range_color=[-5, np.max(ccm_bin.values)],
                     color_continuous_scale="Reds",
                     labels=dict(x="", y="", color=""),
                     title='Binary Co-occurrence Matrix Heatmap')
    
    fig2.update_layout(
        autosize=True,
        title_x=0.5,
        title_font=dict(size=30)
    )

    fig2.update_xaxes(
        tickfont=dict(size=20),
        tickangle=-90,
        ticktext=ccm.columns.tolist(),
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        tickcolor="black",
        showgrid=False
    )

    fig2.update_yaxes(
        tickfont=dict(size=20),
        ticktext=ccm.columns.tolist(),
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        tickcolor="black",
        showgrid=False
    )

    for i in range(num_cols):
        x0 = 0.5 + i
        y1 = num_rows - 0.5

        fig1.add_shape(type="line",
            x0=x0, y0=-0.5,
            x1=x0, y1=y1,
            line=dict(color="white", width=1)
        )
        fig2.add_shape(type="line",
            x0=x0, y0=-0.5,
            x1=x0, y1=y1,
            line=dict(color="white", width=1)
        )

    for i in range(num_rows):
        y0 = 0.5 + i
        x1 = num_cols - 0.5

        fig1.add_shape(type="line",
            x0=-0.5, y0=y0,
            x1=x1, y1=y0,
            line=dict(color="white", width=1)
        )
        fig2.add_shape(type="line",
            x0=-0.5, y0=y0,
            x1=x1, y1=y0,
            line=dict(color="white", width=1)
        )

    fig2.update_xaxes(constrain='domain')
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)

    if show_values is False:
        fig1['data'][0]['texttemplate'] = ''
        fig2['data'][0]['texttemplate'] = ''

    if show is True:
        fig1.show(config={'responsive': True})
        fig2.show(config={'responsive': True})

    return fig1, fig2