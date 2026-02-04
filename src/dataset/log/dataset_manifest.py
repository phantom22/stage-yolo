import pandas as pd
from sklearn.decomposition import PCA

from .backends import *

pca = PCA(n_components=2)

class DatasetManifest:
    def __init__(self, name, df, df_bin, ccm, ccm_bin, 
                instance_counts, photo_occurrence_counts,
                column_ids, column_ids2labels, **kw):
        self.name = name.upper()

        drop = kw.get("drop")
        if drop is None:
            self.raw = df
            self.raw_bin = df_bin
            self.ccm = ccm
            self.ccm_bin = ccm_bin
            self.instance_counts = instance_counts
            self.photo_occurrence_counts = photo_occurrence_counts
            self.column_ids = column_ids
            self.column_ids2labels = column_ids2labels 
            self.dropped = None
        else:
            drop_set = set(drop)
            _cids = [cid for cid in column_ids if cid not in drop_set]
            self.raw = df.loc[_cids]
            self.raw_bin = df_bin.loc[_cids]
            self.ccm = ccm.loc[_cids,_cids]
            self.ccm_bin = ccm_bin.loc[_cids,_cids]
            self.instance_counts = instance_counts[_cids]
            self.photo_occurrence_counts = photo_occurrence_counts[_cids]
            self.column_ids = _cids
            self.column_ids2labels = {k: v for k, v in column_ids2labels.items() if k not in drop_set}
            self.dropped = [column_ids2labels[cid] for cid in drop if cid in column_ids2labels]
            print(f"DATASET:{self.name}, dropped_columns:{self.dropped}")

    def query(self, gt=None, lt=None):
        A = gt is not None
        B = lt is not None

        if not A and not B:
            raise ValueError("No spec rows were passed to the query.")
            
        all_spec_keys = {k for d in [gt, lt] if d for k in d.keys()}
        existing_keys = set(self.raw.index)
        invalid_keys = all_spec_keys - existing_keys

        if invalid_keys:
            raise KeyError(f"The following features were not found in the data index: {invalid_keys}")

        
        if A:
            gt_subset = self.raw.loc[list(gt.keys())]
            gt_met = gt_subset.ge(pd.Series(gt), axis=0).all(axis=0)

        if B:
            lt_subset = self.raw.loc[list(lt.keys())]
            lt_met = lt_subset.le(pd.Series(lt), axis=0).all(axis=0)

        if A and B:
            final_mask = gt_met & lt_met
        elif A:
            final_mask = gt_met
        else:
            final_mask = lt_met
        
        return final_mask[final_mask].index.tolist()

    def prepare_2D_PCA_fig(self, **kw):
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
        
        lib = kw.get("lib", "plotly")
        mode = kw.get("mode", "heatmap")
        show = kw.get("show", True)
        show_values = kw.get("show_values", False)
        fmt = kw.get("fmt", "1.f")

        match lib:
            case "matplotlib":
                match mode:
                    case "heatmap":
                        return prepare_CCM_matplotlib_heatmap_figs(self.name, ccm_renamed, ccm_bin_renamed, show_values, fmt, self.dropped, show)
                    case _:
                        raise Exception(f"prepare_CCM_figs: lib='matplotlib': 'heatmap' is the only suppported mode. got mode='{mode}'")
            case "plotly":
                match mode:
                    case "heatmap":
                        return prepare_CCM_plotly_heatmap_figs(self.name, ccm_renamed, ccm_bin_renamed, show_values, fmt, self.dropped, show)
                    case _:
                        raise Exception(f"prepare_CCM_figs: lib='plotly': 'heatmap' is the only suppported mode. got mode='{mode}'")
            case _:
                raise Exception(f"prepare_CCM_figs: 'plotly' and 'matplotlib' are the only supported libs. got lib='{lib}'.")