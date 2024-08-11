import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo


def main():
    dry_bean = fetch_ucirepo(id=602)

    X = dry_bean.data.features
    y = dry_bean.data.targets

    sfm = SelectFromModel(RandomForestClassifier()).fit(X, y)
    pca = PCA(n_components=2).fit(X)  # 2 components chosen based on Notebook
    le = LabelEncoder().fit(y)

    X_sfm = pd.DataFrame(sfm.transform(X), columns=sfm.get_feature_names_out())
    X_pca = pd.DataFrame(pca.transform(X), columns=pca.get_feature_names_out())
    y_enc = le.transform(y)

    X_sfm['target'] = y_enc
    X_pca['target'] = y_enc

    X_sfm.to_csv('data//sfm.csv', index=None)
    X_pca.to_csv('data//pca.csv', index=None)

    with open('data//le.pkl', 'wb') as f:
        pickle.dump(le, f)


if __name__ == "__main__":
    main()
