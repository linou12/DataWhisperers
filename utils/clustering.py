from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import HDBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def run_hdbscan_grid_search(X_df: pd.DataFrame, min_cluster_size_list: list[int], cluster_selection_epsilon_list: list[float]) -> tuple[np.array, np.array, np.array]:
    """
    Executes a grid search over combinations of `min_cluster_size` and `cluster_selection_epsilon`
    parameters for the HDBSCAN clustering algorithm. Computes and records evaluation metrics (Silhouette
    Score, number of clusters, and number of outliers) for each parameter combination.

    The function returns three matrices containing the silhouette scores, the number of clusters and
    the number of outliers for each combination of the parameters.

    :param X_df: A dataset in a dataframe format to be clustered using HDBSCAN.
    :param min_cluster_size_list: A list specifying potential values for the `min_cluster_size`
                                   parameter of HDBSCAN, which defines the minimum allowable size
                                   of clusters.
    :param cluster_selection_epsilon_list: A list specifying potential values for the
                                            `cluster_selection_epsilon` parameter of HDBSCAN,
                                            which affects cluster merging distance thresholds.
    :return: A tuple of three 2D numpy arrays:
             (1) score_list: Records silhouette scores for each parameter combination.
             (2) nb_clusters_list: Records the number of clusters for each parameter combination.
             (3) nb_outliers_list: Records the number of outliers for each parameter combination.
    """
    # Initialize lists to store the best scores, nb clusters and nb_outliers
    score_list = -1 * np.ones((len(min_cluster_size_list), len(cluster_selection_epsilon_list)))
    nb_clusters_list = -1 * np.ones((len(min_cluster_size_list), len(cluster_selection_epsilon_list)))
    nb_outliers_list = -1 * np.ones((len(min_cluster_size_list), len(cluster_selection_epsilon_list)))

    # Run HDBSCAN for each min_cluster_size and cluster_selection_epsilon
    for id_min, min_cluster_size in enumerate(min_cluster_size_list):
        for id_eps, cluster_selection_epsilon in enumerate(cluster_selection_epsilon_list):

            # Run the algorithm with the current parameters
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)
            labels = clusterer.fit(X_df)

            # Compute the silhouette score, the number of clusters and the number of outliers
            score = silhouette_score(X_df, labels.labels_)
            nb_clusters = len(set(labels.labels_))
            nb_outliers = np.count_nonzero(labels.labels_ == -1)

            # Store the values computed
            score_list[id_min, id_eps] = score
            nb_clusters_list[id_min, id_eps] = nb_clusters
            nb_outliers_list[id_min, id_eps] = nb_outliers

            # Print the current results
            print(f'For min_cluster_size={min_cluster_size} and cluster_selection_epsilon={cluster_selection_epsilon}:')
            print(f'\tNb clusters: {nb_clusters}')
            print(f'\tNb outliers: {nb_outliers}')
            print(f'\tSilhouette score: {score}')
        print('-----------------------------------')

    return score_list, nb_clusters_list, nb_outliers_list

def plot_hyperparameter_tuning_heatmap(score_list: np.array, min_cluster_size_list: list[int], cluster_selection_epsilon_list: list[float], nb_clusters_list: np.array, nb_outliers_list: np.array) -> None:
    """
    Generate a heatmap visualization for hyperparameter tuning results. The heatmap displays the
    relationship between minimum cluster size and cluster selection epsilon, annotated with the number
    of clusters and outliers for each combination of parameters.

    :param score_list: 2D list or array containing silhouette scores for each combination of the
        hyperparameter values. Values represent how well the clusters are formed for
        specific parameter combinations.
    :param min_cluster_size_list: List of minimum cluster sizes corresponding to the rows of the heatmap.
        Each value denotes a specific value of the minimum cluster size hyperparameter.
    :param cluster_selection_epsilon_list: List of cluster selection epsilon values corresponding to the
        columns of the heatmap. Each value represents a specific cluster selection epsilon
        used during tuning.
    :param nb_clusters_list: 2D list or array representing the number of identified clusters for each
        combination of hyperparameter values. Each element corresponds to the associated combination
        of minimum cluster size and cluster selection epsilon.
    :param nb_outliers_list: 2D list or array representing the number of outliers detected for each
        combination of hyperparameter values. Each element corresponds to the associated
        combination of the minimum cluster size and cluster selection epsilon.
    :return: None
    """

    plt.figure(figsize=(10, 8))
    plt.imshow(score_list, aspect='auto')
    plt.colorbar(label='Silhouette Score')

    # Set ticks and labels
    plt.xticks(range(len(cluster_selection_epsilon_list)), cluster_selection_epsilon_list)
    plt.yticks(range(len(min_cluster_size_list)), min_cluster_size_list)

    # Add text annotations to each cell
    for i in range(len(min_cluster_size_list)):
        for j in range(len(cluster_selection_epsilon_list)):
            plt.text(j, i, f'Nb clusters:{nb_clusters_list[i, j]:.0f}\nNb outliers:{nb_outliers_list[i, j]:.0f}',
                     ha='center', va='center', color='white')

    plt.xlabel('Cluster Selection Epsilon')
    plt.ylabel('Min Cluster Size')
    plt.title('Hyperparameter Tuning Heatmap')

    plt.tight_layout()
    plt.show()


def init_fit_hdbscan_clustering(x: pd.DataFrame, min_cluster_size: int = 5, cluster_selection_epsilon: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Initializes an HDBSCAN clusterer and fits it to the given dataset by appending cluster labels and probabilities to the dataset.
    Additionally, returns the dataset augmented with clustering information and the
    centroid positions of the identified clusters.

    :param x: Input dataset as a DataFrame where observations are represented as rows
        and features as columns.
    :param min_cluster_size: The minimum size of clusters for HDBSCAN clustering.
    :param cluster_selection_epsilon: The distance threshold for merging clusters
        during cluster selection.
    :return: A tuple where the first element is a DataFrame augmented with labels and
        probabilities for each observation, and the second element is a DataFrame
        containing the centroids of the identified clusters.
    """
    # First initialize the clusterer and fit on the data
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, store_centers='centroid', cluster_selection_epsilon=cluster_selection_epsilon)
    labels = clusterer.fit(x)

    print(f'Model fitted. Num of clusters found: {len(labels.centroids_)}')

    # Rename columns of centroids
    y_centroids = pd.DataFrame(labels.centroids_)
    y_centroids.columns = x.columns

    # Append the labels and probabilities to the dataset
    y = x.copy()
    y['labels'] = labels.labels_
    y['probabilities'] = labels.probabilities_

    print('Labels and probabilities added')

    return y, y_centroids

def create_radar_chart(df, figsize=(10, 10)):
    """
    Create a radar chart from a DataFrame where each row is plotted as a separate layer
    and columns represent the dimensions.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame where each row will be a layer and columns are dimensions
    figsize : tuple
        Figure size as (width, height)
    """
    # Get the categories/columns and number of variables
    categories = df.columns
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close the plot by appending the first angle
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    # Plot data for each row
    for idx, row in df.iterrows():
        # Complete the loop by appending first value
        values = row.values.tolist()
        values += values[:1]

        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'Row {idx}')
        ax.fill(angles, values, alpha=0.1)

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Add title
    plt.legend(loc='best')
    plt.title("Radar Chart")

    plt.tight_layout()
    plt.show()

def grid_search_GMM(df: pd.DataFrame, components_list: list[int], num_iter: int = 4):
    """
    Perform a grid search for the best number of components in a Gaussian Mixture Model
    (GMM) by evaluating the average silhouette score for multiple iterations.

    :param df: Dataset on which clustering will be performed. Should be a 2D array-like
        structure with samples as rows and features as columns.
    :param components_list: List of integers specifying the number of components to
        evaluate in the GMM.
    :param num_iter: Number of iterations to compute silhouette scores for stability.
        Defaults to 4 if not specified.
    """
    sil_scores = []
    for n_comp in components_list:
        gm = GaussianMixture(n_components=n_comp).fit(df)
        labels = gm.predict(df)
        score = np.mean([silhouette_score(df, labels) for _ in range(num_iter)])
        sil_scores.append(score)
        print(f'n_comp: {n_comp}, score: {score}')

    plt.plot(components_list, sil_scores)
    plt.xlabel('Number of components')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score by number of components')
    plt.show()

def fit_and_plot_GMM(df: pd.DataFrame, n_comp: int):
    """
    Fits a Gaussian Mixture Model (GMM) to the provided DataFrame and creates a radar
    chart visualization based on the cluster centers. The function then prints the
    counts of data points in each cluster determined by the GMM.

    :param df: The input DataFrame containing data points to fit the Gaussian Mixture Model.
    :param n_comp: The number of clusters to initialize in the Gaussian Mixture Model.
    """
    gm = GaussianMixture(n_components=n_comp).fit(df)
    clusters = pd.DataFrame(gm.means_)
    clusters.columns = df.columns
    create_radar_chart(clusters)
    print(pd.DataFrame(np.unique_counts(gm.predict(df)).counts))