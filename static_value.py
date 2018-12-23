import seaborn as sns
import matplotlib.pyplot as plt

train_LEN = 7377418
test_LEN = 2556790
proportionment = test_LEN/(test_LEN + train_LEN)
test_data_len = test_LEN


def plot_correlation_map(df):

    corr = df.corr()
    plt.yticks(rotation=60)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        # ax=ax,
        annot=True,
        annot_kws={'fontsize': 6}

    )

    plt.xticks(rotation=60)
    plt.show()

def chane_property_importance_number(dfone):
    d = dfone.value_counts()
    return d[dfone.values].values