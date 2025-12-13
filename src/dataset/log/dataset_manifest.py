import pandas as pd
from sklearn.decomposition import PCA

from .modes.prepare_2D_PCA_fig.plotly import fig as prepare_2D_PCA_plotly_fig
from .modes.prepare_2D_PCA_fig.mathplotlib import fig as prepare_2D_PCA_mathplotlib_fig
from .modes.prepare_CCM_figs.mathplotlib import fig as prepare_CCM_mathplotlib_figs

class DatasetManifest:
    def __init__(self, name, og, vec, column_ids, column_ids2labels):
        self.name = name.upper()
        self.og = og
        self.vec = vec
        self.column_ids = column_ids
        self.column_ids2labels = column_ids2labels 

        raw = pd.DataFrame(vec, columns=column_ids)
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

        mode = kw.get("mode","plotly")
        show = kw.get("show",True)

        match mode:
            case "plotly":
                fig = prepare_2D_PCA_plotly_fig(self.name, pca_df, pc_data_list, show)
            case "mathplotlib":
                fig = prepare_2D_PCA_mathplotlib_fig(self.name, pca_df, pc_data_list, show)
            case _:
                raise Exception(f"prepare_2D_PCA_fig: 'plotly' and 'mathplotlib' are the only supported modes. got '{mode}'.")

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
        
        mode = kw.get("mode", "mathplotlib")
        show = kw.get("show", True)
        show_values = kw.get("show_values", False)
        fmt = kw.get("fmt", "1.f")


        match mode:
            case "mathplotlib":
                return prepare_CCM_mathplotlib_figs(ccm_renamed, ccm_bin_renamed, show_values, fmt, show)
            case _:
                raise Exception(f"prepare_CCM_figs: 'mathplotlib' is the only supported mode. got '{mode}'.")