import plotly.express as px
import numpy as np
import plotly.graph_objects as go

def prepare_CCM_plotly_heatmap_figs(name, ccm, ccm_bin, show_values, fmt, dropped, show):
    num_rows, num_cols = ccm.shape
    show_legend = dropped is not None

    ccm_masked = ccm.replace(0, np.nan)

    fig1 = px.imshow(ccm_masked,
                     text_auto=fmt,
                     aspect="auto",
                     range_color=[0, np.max(ccm_masked.values)],
                     color_continuous_scale="Reds",
                     labels=dict(x="", y="", color=""),
                     title=f'{name} Co-occurrence Matrix Heatmap')
    
    fig1.update_layout(
        autosize=True,
        title_x=0.5,
        title_font=dict(size=30),
        showlegend=show_legend,
        legend=dict(
            x=0,
            y=1,
            font=dict(size=28),
            xanchor='left',
            yanchor='top',
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        plot_bgcolor="rgba(0,0,0,0.2)"
    )

    fig1.update_xaxes(
        tickfont=dict(size=20),
        tickangle=-90,
        ticktext=ccm.columns,
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        tickcolor="black",
        showgrid=False
    )

    fig1.update_yaxes(
        tickfont=dict(size=20),
        ticktext=ccm.columns,
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        tickcolor="black",
        showgrid=False
    )

    fig1.update_xaxes(constrain='domain')
    fig1.update_yaxes(scaleanchor="x", scaleratio=1)

    ccm_bin_masked = ccm_bin.replace(0, np.nan)

    fig2 = px.imshow(ccm_bin_masked,
                     text_auto=fmt,
                     aspect="auto",
                     range_color=[0, np.max(ccm_bin_masked.values)],
                     color_continuous_scale="Reds",
                     labels=dict(x="", y="", color=""),
                     title=f'{name} Binary Co-occurrence Matrix Heatmap')
    
    fig2.update_layout(
        autosize=True,
        title_x=0.5,
        title_font=dict(size=30),
        showlegend=show_legend,
        legend=dict(
            x=0,
            y=1,
            font=dict(size=28),
            xanchor='left',
            yanchor='top',
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        plot_bgcolor="rgba(0,0,0,0.2)"
    )

    fig2.update_xaxes(
        tickfont=dict(size=20),
        tickangle=-90,
        ticktext=ccm.columns,
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        tickcolor="black",
        showgrid=False
    )

    fig2.update_yaxes(
        tickfont=dict(size=20),
        ticktext=ccm.columns,
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

    if show_legend:
        trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name='<b>EXCLUDED CLASSES:</b>',
            marker=dict(color='rgba(0,0,0,0)'), # Invisible
            showlegend=True
        )

        fig1.add_trace(trace)
        fig2.add_trace(trace)

        # 3. Loop through your self.dropped list and add them
        for item in dropped:
            trace = go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name=f" - {item}",
                marker=dict(color='rgba(0,0,0,0)'), # Invisible
                showlegend=True
            )

            fig1.add_trace(trace)
            fig2.add_trace(trace)

    fig2.update_xaxes(constrain='domain')
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)

    if show_values is False:
        fig1['data'][0]['texttemplate'] = ''
        fig2['data'][0]['texttemplate'] = ''

    if show is True:
        fig1.show(config={'responsive': True})
        fig2.show(config={'responsive': True})

    return fig1, fig2