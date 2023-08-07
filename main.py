import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
from matplotlib import pyplot as plt
from factor_analyzer import FactorAnalyzer
from matplotlib import gridspec
from options import *


def columnization(df, names):
    """
    Determines the column in the matrix to analyze

    Parameters:
        df: pandas DataFrame
        names: list on column names

    Returns:
        (column names of participant's data, column names for blood mother, column names for foster mother)
    """
    columns_limit_l = np.argwhere(df.columns.values == names[0])[0][0] + 1
    columns_limit_r = np.argwhere(df.columns.values == names[1])[0][0] - 1
    columns_limit_l_m1 = np.argwhere(df.columns.values == names[2])[0][0]
    columns_limit_r_m1 = np.argwhere(df.columns.values == names[3])[0][0] + 1
    columns_limit_l_m2 = np.argwhere(df.columns.values == names[4])[0][0]
    columns_limit_r_m2 = np.argwhere(df.columns.values == [5])[0][0] + 1

    compare_col = df.columns.values[columns_limit_l:columns_limit_r]
    col_for_heat_1 = list(range(columns_limit_l, columns_limit_r + 1)) \
                     + list(range(columns_limit_l_m1, columns_limit_r_m1))
    col_for_heat_2 = list(range(columns_limit_l, columns_limit_r + 1)) \
                     + list(range(columns_limit_l_m2, columns_limit_r_m2))

    return compare_col, col_for_heat_1, col_for_heat_2


def heatmap_clustered(df, filename, colors, columns, threshold=0.8, show_plot=True, pca=False, factors=3):
    """
    Performs ???

    Parameters:
        df: pandas DataFrame
        filename: suffix in output filename
        colors: color palette for variable categories
        columns: columns from df to choose
        threshold: threshold for clustering
        show_plot: if plot must be shown
        pca: if factor analysis must be performed
        factors: numbers of factors to consider

    Returns:
        loadings: loadings of factors, if pca == True
        total_variance: cumulative variance, if pca == True

    """
    df_heatmap = df.iloc[:, columns].copy()
    correlations = df_heatmap.corr(method='pearson')

    g = sns.clustermap(abs(round(correlations, 2)),
                       method="complete",
                       cmap='coolwarm',
                       annot=True,
                       dendrogram_ratio=(.1, .4),
                       row_cluster=True,
                       col_cluster=True,
                       col_colors=colors,
                       cbar_pos=None,  # (.829, .832, .158, .019),
                       cbar_kws={"orientation": "vertical"},
                       annot_kws={"size": 7},
                       vmin=-1,
                       vmax=1,
                       figsize=(15, 12))
    legend_TN = [mpatches.Patch(color=c, label=l)
                 for c, l in zip(['blue', 'green', 'orange', 'purple', 'yellow', 'brown', 'gray', 'black'],
                                 ['Compulsory', 'Locomotion', 'Anxiety', 'Social',
                                  'Novelty', 'Depression', 'Exploratory', "Maternal"])]
    l2 = g.ax_heatmap.legend(loc='lower right', bbox_to_anchor=(0.0, 1.1), handles=legend_TN, frameon=True)
    l2.set_title(title='Variable group', prop={'size': 10})
    plt.savefig("./Graphs/" + filename + "cluster.jpg")
    if show_plot:
        plt.show()

    dissimilarity = 1 - abs(round(correlations, 2))
    z_linkage = linkage(squareform(dissimilarity), 'complete')
    labels = fcluster(z_linkage, threshold, criterion='distance')
    labels_order = np.argsort(labels)

    for idx, i in enumerate(df_heatmap.columns[labels_order]):
        if idx == 0:
            clustered = pd.DataFrame(df_heatmap[i])
        else:
            df_to_append = pd.DataFrame(df_heatmap[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)

    correlations = clustered.corr()

    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(round(correlations, 2), cmap='coolwarm', annot=True,
                annot_kws={"size": 6}, vmin=-1, vmax=1, ax=ax)
    fig.tight_layout()
    plt.savefig("./Graphs/" + filename + "heatmap.jpg")
    if show_plot:
        plt.show()
    plt.close(fig=fig)

    if pca:
        fa = FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=True,
                            method='minres', n_factors=factors, rotation='varimax', rotation_kwargs={},
                            use_smc=True)
        fa.fit(correlations)

        plt.close('all')
        ev = fa.get_eigenvalues()[0]
        plt.plot(ev)
        plt.show()

        loadings = pd.DataFrame(fa.loadings_,
                                index=clustered.columns)

        total_variance = fa.get_factor_variance()[2]

        return loadings, total_variance


def mann_whitney_all_levels(df, factor, columns):
    """
    Performs pair-wise Mann-Whithey comparison between all combination of factor levels in set of variables

    Parameters:
        df: pandas DataFrame
        factor: column name to split subjects in all combinations of levels
        columns: list of column names with variables

    Returns:
        out_df: pandas DataFrame with summary

    """
    grps = df[factor].unique()
    compare_df = pd.DataFrame(columns=[f'{c1}_{c2}' for c1, c2 in combinations(grps, 2)])

    for var_MW in columns:
        tests = {
            f'{c1}_{c2}': stats.mannwhitneyu(
                df.loc[df[factor] == c1, var_MW],
                df.loc[df[factor] == c2, var_MW]
            ) for c1, c2 in combinations(grps, 2)
        }
        compare_df = pd.concat([compare_df,
                                pd.DataFrame(tests, index=[var_MW + '_T', var_MW + '_p']).iloc[[1], :]])

    out_df = compare_df.where(compare_df < .05, np.nan)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(out_df,
                linewidths=1,
                linecolor='gray',
                cmap=sns.color_palette("flare_r", as_cmap=True),
                ax=ax)
    plt.title('Mann-Whitney pair-wise comparison p-values')
    plt.savefig("./Graphs/mann-whitney_heatmap.jpg")
    plt.show()

    return out_df


def anova_with_graph(df, f, compare_col, labels, plot=False, show_plot=False, suffix=''):
    """
    Performs pair-wise Mann-Whithey comparison between all combination of factor levels in set of variables

    Parameters:
        df: pandas DataFrame
        f:
        compare_col:
        labels:
        plot:
        show_plot:
        suffix:

    Returns:
        anova_df: pandas DataFrame with summary

    """
    df = df.copy()
    df.reset_index(inplace=True)
    anova_df = pd.DataFrame(columns=['F', 'PR(>F)'])

    for var_plot_anova in compare_col:
        try:
            model = ols(f'{var_plot_anova} ~ C({f[0]}) + C({f[1]}) + C({f[0]}):C({f[1]})',
                        data=df).fit()
            k = sm.stats.anova_lm(model, typ=2)
            k['Var'] = [var_plot_anova, var_plot_anova, var_plot_anova, var_plot_anova]
            anova_df = pd.concat([anova_df, k.iloc[:3, 2:]])

            if plot:
                fig, ax = plt.subplots(figsize=(8, 8))
                interaction_plot(
                    x=df[f[0]],
                    trace=df[f[1]],
                    response=df[var_plot_anova],
                    colors=["red", "blue"],
                    markers=["D", "^"],
                    ms=10,
                    ax=ax,
                    xlabel=labels[0],
                    legendtitle=labels[1]
                )
                plt.savefig('./Graphs/anova_' + var_plot_anova + suffix + '.jpg')
                if show_plot:
                    plt.show()
                plt.close()
        except:
            plt.close('all')
            print(f'{var_plot_anova} omitted!')

    anova_df['PR(>F)'].where(anova_df['PR(>F)'] < .05, np.nan, inplace=True)
    anova_df.dropna(inplace=True)

    return anova_df


def sort_factanal(df, base=10, threshold=0.5, plot=True, suffix=''):
    """
    Performs ???

    Parameters:
        df: pandas DataFrame
        base: base for exponential sorting coefficients
        threshold: correlation coefficient threshold to ignore in sorting
        plot: if plot must be shown
        suffix: suffix in the filename

    Returns:
        df: sorted DataFrame with factor loadings

    """
    multiplier = {key: 1 / (base ** x) for key, x in zip(df.columns, range(1, len(df.columns) + 1))}
    df_temp = abs(df)
    df_temp.mask(df_temp < threshold, np.NaN, inplace=True)
    df_temp = df_temp.mul(multiplier, axis='columns')
    sums = df_temp.sum(axis=1, skipna=True)
    sums.sort_values(inplace=True, ascending=False)
    df = df.loc[sums.index, :]

    if plot:
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(1, len(df.columns) + 1, width_ratios=[1] + [4] * len(df.columns))
        axes = [plt.subplot(gs[i]) for i in range(len(df.columns) + 1)]
        y_pos = np.arange(len(df))

        colors_bar = [cmaps.get(sp, 'black') for sp in df.index]
        sns.barplot(x=[1] * len(y_pos), y=y_pos, ax=axes[0], orient='h', palette=colors_bar)
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel('')
        axes[0].get_xaxis().set_visible(False)
        axes[0].set_yticks(y_pos, labels=df.index)

        for i in range(1, len(df.columns) + 1):
            colors = ['red' if x > 0 else 'blue' for x in df[df.columns[i - 1]]]
            sns.barplot(x=abs(df[df.columns[i - 1]]), y=y_pos, ax=axes[i], orient='h', palette=colors)
            axes[i].set_title(df.columns[i - 1])
            axes[i].set_xlim(0, 1)
            axes[i].set_xlabel('')
            axes[i].get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.savefig('./Graphs/factors_' + suffix + '.jpg')
        plt.show()
        plt.close('all')

    return df


if __name__ == '__main__':
    heat_maps = True
    anova_graph = False
    df_raw = pd.read_csv('./Input/Maryia-all-fix.csv')
    df_raw.columns = df_raw.columns.str.replace(".", "_")
    df_raw.drop(columns=DROP_COLUMNS, inplace=True, errors='ignore')
    #    df_raw.dropna(inplace=True)

    compare_columns, col_for_heatmap_1, col_for_heatmap_2 = columnization(df_raw,
                                                                          names=['Group',
                                                                                 'Treathment_x',
                                                                                 'Latency_carrying_x',
                                                                                 'Retrieval_interval_x',
                                                                                 'Latency_carrying_y',
                                                                                 'Retrieval_interval_y'])

    all_levels_tests = mann_whitney_all_levels(df=df_raw,
                                               factor='Group',
                                               columns=compare_columns)
    cmaps = CMAPS

    if heat_maps:
        heatmap_clustered(df_raw,
                          'all_',
                          colors=[cmaps.get(sp, 'black') for sp in df_raw.columns.values[col_for_heatmap_2]],
                          columns=col_for_heatmap_2,
                          show_plot=False)
        heatmap_clustered(df_raw.loc[df_raw['Sex'] == 'm',],
                          'all_male_',
                          colors=[cmaps.get(sp, 'black') for sp in df_raw.columns.values[col_for_heatmap_2]],
                          columns=col_for_heatmap_2,
                          show_plot=False)
        heatmap_clustered(df_raw.loc[df_raw['Sex'] == 'f',],
                          'all_female_',
                          colors=[cmaps.get(sp, 'black') for sp in df_raw.columns.values[col_for_heatmap_2]],
                          columns=col_for_heatmap_2,
                          show_plot=False)
        heatmap_clustered(df_raw.loc[df_raw['Treathment_x'] == 'PRX',],
                          'all_treated_',
                          colors=[cmaps.get(sp, 'black') for sp in df_raw.columns.values[col_for_heatmap_2]],
                          columns=col_for_heatmap_2,
                          show_plot=False)
        heatmap_clustered(df_raw.loc[df_raw['Treathment_x'] == 'CTR',],
                          'all_controls_',
                          colors=[cmaps.get(sp, 'black') for sp in df_raw.columns.values[col_for_heatmap_2]],
                          columns=col_for_heatmap_2,
                          show_plot=False)

        df_cf = df_raw.loc[df_raw['CF_x'] == 'CF', :].copy()
        df_ncf = df_raw.loc[df_raw['CF_x'] == 'nonCF', :].copy()

        heatmap_clustered(df_cf.loc[df_cf['Treathment_x'] == 'PRX',],
                          'CF_PRX_blood_',
                          colors=[cmaps.get(sp, 'black') for sp in df_raw.columns.values[col_for_heatmap_1]],
                          columns=col_for_heatmap_1,
                          show_plot=False)
        heatmap_clustered(df_cf.loc[df_cf['Treathment_x'] == 'CTR',],
                          'CF_CRT_blood_',
                          colors=[cmaps.get(sp, 'black') for sp in df_raw.columns.values[col_for_heatmap_1]],
                          columns=col_for_heatmap_1,
                          show_plot=False)

        loadings_pca_PRX_cf, var_PRX_cf = heatmap_clustered(df_cf.loc[df_cf['Treathment_x'] == 'PRX',],
                                                            'CF_PRX_foster_',
                                                            colors=[cmaps.get(sp, 'black') for sp in
                                                                    df_raw.columns.values[col_for_heatmap_2]],
                                                            columns=col_for_heatmap_2,
                                                            show_plot=False,
                                                            pca=True)

        loadings_pca_CRT_cf, var_CRT_cf = heatmap_clustered(df_cf.loc[df_cf['Treathment_x'] == 'CTR',],
                                                            'CF_CRT_foster_',
                                                            colors=[cmaps.get(sp, 'black') for sp in
                                                                    df_raw.columns.values[col_for_heatmap_2]],
                                                            columns=col_for_heatmap_2,
                                                            show_plot=False,
                                                            pca=True)

        loadings_pca_PRX_ncf, var_PRX_ncf = heatmap_clustered(df_ncf.loc[df_ncf['Treathment_x'] == 'PRX',],
                                                              'nonCF_PRX_foster_',
                                                              colors=[cmaps.get(sp, 'black') for sp in
                                                                      df_raw.columns.values[col_for_heatmap_2]],
                                                              columns=col_for_heatmap_2,
                                                              factors=4,
                                                              show_plot=False,
                                                              pca=True)

        loadings_pca_CRT_ncf, var_CRT_ncf = heatmap_clustered(df_ncf.loc[df_ncf['Treathment_x'] == 'CTR',],
                                                              'nonCF_CRT_foster_',
                                                              colors=[cmaps.get(sp, 'black') for sp in
                                                                      df_raw.columns.values[col_for_heatmap_2]],
                                                              columns=col_for_heatmap_2,
                                                              factors=5,
                                                              show_plot=False,
                                                              pca=True)

        loadings_pca_PRX_cf = sort_factanal(loadings_pca_PRX_cf, threshold=0.4, suffix='PRX_cf')
        loadings_pca_CRT_cf = sort_factanal(loadings_pca_CRT_cf, threshold=0.4, suffix='CRT_cf')
        loadings_pca_PRX_ncf = sort_factanal(loadings_pca_PRX_ncf, threshold=0.4, suffix='PRX_ncf')
        loadings_pca_CRT_ncf = sort_factanal(loadings_pca_CRT_ncf, threshold=0.4, suffix='CRT_ncf')

    if anova_graph:
        anova_treat_vs_CF = anova_with_graph(df=df_raw,
                                             compare_col=compare_columns,
                                             f=['Treathment_x', 'CF_x'],
                                             labels=['Treatment', 'Cross-fostering'],
                                             plot=True, show_plot=False, suffix='_TvCF')
        anova_treat_vs_foster = anova_with_graph(df=df_raw,
                                                 compare_col=compare_columns,
                                                 f=['Treathment_x', 'Treathment_y'],
                                                 labels=['Treatment_embryo', 'Treatment_nurture'],
                                                 plot=True, show_plot=False, suffix='_TvT')
        anova_CF_vs_foster = anova_with_graph(df=df_raw,
                                              f=['CF_x', 'Treathment_y'],
                                              compare_col=compare_columns,
                                              labels=['Cross-fostering', 'Treatment_nurture'],
                                              plot=True, show_plot=False, suffix='_CFvT')
        anova_treat_vs_sex = anova_with_graph(df=df_raw,
                                              compare_col=compare_columns,
                                              f=['Treathment_x', 'Sex'],
                                              labels=['Treatment', 'Sex'],
                                              plot=True, show_plot=False, suffix='_TvSex')
