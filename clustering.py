"""Ward hierarchical clustering directly on fitted SVI parameters -- the PCA
replacement for finding recurring smile/skew regimes.
"""

from scipy.cluster.hierarchy import fcluster, linkage

FEATURES = ["a", "b", "rho", "m", "sigma"]


def cluster_svi_params(cleaned_panel, n_clusters=5):
    X = cleaned_panel[FEATURES].to_numpy()
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    link = linkage(Z, method="ward")
    labels = fcluster(link, t=n_clusters, criterion="maxclust")

    out = cleaned_panel.copy()
    out["cluster"] = labels
    return out, link


def characterize_clusters(clustered_panel):
    summary = clustered_panel.groupby("cluster")[FEATURES].mean()
    summary["n_slices"] = clustered_panel.groupby("cluster").size()

    sigma_median = summary["sigma"].median()

    def tag(row):
        if row["rho"] < -0.3:
            skew = "put-skew-steep"
        elif row["rho"] > 0.3:
            skew = "call-skew"
        else:
            skew = "near-symmetric"
        wing = "fat-wings" if row["sigma"] > sigma_median else "thin-wings"
        return f"{skew}/{wing}"

    summary["regime_tag"] = summary.apply(tag, axis=1)
    return summary
