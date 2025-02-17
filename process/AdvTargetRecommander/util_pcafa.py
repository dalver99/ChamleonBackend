import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import ast

def preprocessing_data(ad_gt):
    labels = ['male', 'female', '10gen', '20gen', '30gen', '40gen', '50gen', '60gen', 'gaze_num']
    data_pre = pd.DataFrame(columns=labels)
    data_pre['gaze_num'] = ad_gt['gazed_num']
    data_pre.fillna(0, inplace=True)
    data_pre = data_pre.astype('int')

    #Count Gender
    for idx, row in ad_gt.iterrows():
        gender = row['pre_gender']
        group = row['age_group']
        # 이전 값을 유지하고, gender와 group 컬럼에만 +1
        data_pre.loc[idx, :] = data_pre.iloc[idx, :].copy()  # 이전 값 복사 (경고 방지)
        data_pre.loc[idx, gender] = 1
        data_pre.loc[idx, group] = 1

    labels_cum = []
    for col in labels[:-1]:
        labels_cum.append(f'{col}_cum')
    data_pre[labels_cum] = data_pre[labels[:-1]].cumsum()
    labels_cum.append('gaze_num')
    data_pre = data_pre[labels_cum]
    return data_pre

def apply_PCA(data_pre):
    scaler = StandardScaler()
    data_pre_ = scaler.fit_transform(data_pre)
    pca = PCA()
    pca_r= pca.fit(data_pre_)
    x_pca = pca.transform(data_pre_)
    cols = [f'PC{i}' for i in range(1, len(data_pre.T)+1)]
    pca_df = pd.DataFrame(x_pca, columns=cols)
    ev = pca_r.explained_variance_ #ev: explained_variance
    evr = ev/np.sum(ev) #evr: explained variance ratio
    cumm_evr = []
    a = 0
    for i in evr:
        a += i
        cumm_evr.append(float(a))
        if a > 0.7:
            break
    return len(cumm_evr), a, data_pre_, pca_df

# 요인분석
def Apply_FA(n_comp, threshold, data_pre_):
    labels = ['male', 'female', '10gen', '20gen', '30gen', '40gen', '50gen', '60gen', 'gaze_num']
    fa = FactorAnalysis(n_components=n_comp, rotation='varimax')
    fa.fit(data_pre_)
    comp = fa.components_
    fa_df = pd.DataFrame(comp.T, columns=[f"PC{i}" for i in range(1, len(comp)+1)], index=labels)
    temp = []
    pc_n = fa.n_components
    cols = [f'PC{i}' for i in range(1, pc_n+1)]
    for i, row in fa_df.iterrows():
        point1 = row[cols].values
        point2 = fa_df.loc['gaze_num', cols].values
        temp.append(euclidean(point1, point2))
    fa_df['distance'] = temp
    recommand_target_df = fa_df[fa_df['distance'] < threshold]
    return recommand_target_df, fa_df

def fa_pairplot(fa_df, option=False):
    if option == True:
        plot_df = fa_df.reset_index().rename(columns={'index': 'category'})
        cols = fa_df.columns
        vars = [i for i in cols if 'PC' in i]
        colors = sns.color_palette("husl", n_colors=len(fa_df.index))
        color_dict = dict(zip(fa_df.index, colors))

        sns.pairplot(plot_df,
                    x_vars=vars,
                    y_vars=vars,
                    height=2.5,
                    diag_kind='kde',
                    hue='category',
                    palette=color_dict)

        # 제목 추가
        plt.suptitle('Pairplot of Principal Components', y=1.02)

        # 그래프 표시
        plt.show()