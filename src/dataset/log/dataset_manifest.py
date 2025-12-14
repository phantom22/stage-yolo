import pandas as pd
from sklearn.decomposition import PCA

from .backends import *

class DatasetManifest:
    def __init__(self, name, og, vec, column_ids, column_ids2labels, **kw):
        drop = kw.get("drop")
        
        self.name = name.upper()
        self.og = og

        if drop is None:
            _vec = vec
            _column_ids = column_ids
            _column_ids2labels = column_ids2labels
            self.dropped = []
        else:
            drop_set = set(drop)
            _vec = [
                {k: v for k, v in row.items() if k not in drop_set} 
                for row in vec
            ]
            _column_ids = [col_id for col_id in column_ids if col_id not in drop_set]
            _column_ids2labels = {k: v for k, v in column_ids2labels.items() if k not in drop_set}
            self.dropped = [column_ids2labels[col_id] for col_id in drop if col_id in column_ids2labels]
        
        self.vec = _vec
        self.column_ids = _column_ids
        self.column_ids2labels = _column_ids2labels 

        raw = pd.DataFrame(_vec, columns=_column_ids)

        raw_bin = raw.map(lambda x: 1 if x > 0 else 0)

        self.raw = raw.T
        self.raw_bin = raw_bin.T
        self.ccm = raw.T.dot(raw)
        self.ccm_bin = raw_bin.T.dot(raw_bin)

        self.instance_counts = raw.sum(axis=0).values
        self.photo_occurrence_counts = raw_bin.sum(axis=0).values

    def prepare_2D_PCA_fig(self, **kw):
        pca = PCA(n_components=2)

        c_instance_count = pca.fit_transform(self.raw)
        c_photo_occurence_count = pca.fit_transform(self.raw_bin)
        c_co_intensity = pca.fit_transform(self.ccm)
        c_co_presence = pca.fit_transform(self.ccm_bin)

        pca_df = pd.DataFrame({
            'id': self.column_ids,
            'PC1': c_instance_count[:,0],
            'PC2': c_instance_count[:,1],
            'instance_count': self.instance_counts
        })

        pca_df['label'] = pca_df['id'].map(self.column_ids2labels)
        pca_df['photo_occurrence_count'] = self.photo_occurrence_counts

        # Define the PCA data for each subplot
        pc_data_list = [
            ('Instance counts', c_instance_count[:, 0], c_instance_count[:, 1]),
            ('Photo Occurrence', c_photo_occurence_count[:, 0], c_photo_occurence_count[:, 1]),
            ('Co-Intensity', c_co_intensity[:, 0], c_co_intensity[:, 1]),
            ('Co-Presence', c_co_presence[:, 0], c_co_presence[:, 1])
        ]

        lib = kw.get("lib","plotly")
        show = kw.get("show",True)

        match lib:
            case "plotly":
                fig = prepare_2D_PCA_plotly_fig(self.name, pca_df, pc_data_list, show)
            case "matplotlib":
                fig = prepare_2D_PCA_mathplotlib_fig(self.name, pca_df, pc_data_list, show)
            case _:
                raise Exception(f"prepare_2D_PCA_fig: 'plotly' and 'matplotlib' are the only supported modes. got lib='{lib}'.")

        max_photo_occurrence_row = pca_df.loc[pca_df['photo_occurrence_count'].idxmax()]
        min_photo_occurrence_row = pca_df.loc[pca_df['photo_occurrence_count'].idxmin()]

        max_instance_count_row = pca_df.loc[pca_df['instance_count'].idxmax()]
        min_instance_count_row = pca_df.loc[pca_df['instance_count'].idxmin()]

        print(f"[{self.name}]")
        print(f"instance_count max({max_instance_count_row['label']},{max_instance_count_row['instance_count']}), min({min_instance_count_row['label']},{min_instance_count_row['instance_count']})")
        print(f"photo_occurrence max({max_photo_occurrence_row['label']},{max_photo_occurrence_row['photo_occurrence_count']}), min({min_photo_occurrence_row['label']},{min_photo_occurrence_row['photo_occurrence_count']})")

        return fig
    
    def prepare_CCM_figs(self, **kw):
        ccm_renamed = self.ccm.rename(
            index=self.column_ids2labels,
            columns=self.column_ids2labels
        )

        ccm_bin_renamed = self.ccm_bin.rename(
            index=self.column_ids2labels,
            columns=self.column_ids2labels
        )
        
        lib = kw.get("lib", "matplotlib")
        mode = kw.get("mode", "heatmap")
        show = kw.get("show", True)
        show_values = kw.get("show_values", False)
        fmt = kw.get("fmt", "1.f")

        match lib:
            case "matplotlib":
                match mode:
                    case "heatmap":
                        return prepare_CCM_matplotlib_heatmap_figs(ccm_renamed, ccm_bin_renamed, show_values, fmt, show)
                    case _:
                        raise Exception(f"prepare_CCM_figs: lib='matplotlib': 'heatmap' is the only suppported mode. got mode='{mode}'")
            case "plotly":
                match mode:
                    case "heatmap":
                        return prepare_CCM_plotly_heatmap_figs(ccm_renamed, ccm_bin_renamed, show_values, fmt, show)
                    case _:
                        raise Exception(f"prepare_CCM_figs: lib='plotly': 'heatmap' is the only suppported mode. got mode='{mode}'")
            case _:
                raise Exception(f"prepare_CCM_figs: 'plotly' and 'matplotlib' are the only supported libs. got lib='{lib}'.")