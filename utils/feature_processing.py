from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd


def remove_low_variance_feature(df, threshold=0.01):
    """
    take a dataframe with all features and remove the one with low variance
    :param df: features dataframe
    :param threshold: threshold to remove the feature
    :return: dataframe with only the features with high variance
    """
    selector = VarianceThreshold(threshold=threshold)
    features_selected = selector.fit_transform(df)
    columns = df.columns[selector.get_support()]
    return pd.DataFrame(features_selected, columns=columns)


def scale_features(df):
    """
    standardize features
    :param df: dataframe with all features
    :return: standardized dataframe
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    return pd.DataFrame(scaled_features, columns=df.columns)


def process_features(df, threshold=0.01):
    """
    remove low variance features and scale features
    :param df: dataframe with all features
    :param threshold: threshold to remove the feature
    :return: processed dataframe ready for clustering
    """
    df = remove_low_variance_feature(df, threshold=threshold)
    df = scale_features(df)
    return df
