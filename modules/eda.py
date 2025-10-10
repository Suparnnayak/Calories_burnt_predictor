import matplotlib.pyplot as plt
import seaborn as sb

def scatter_plots(df, features, target='Calories'):
    plt.subplots(figsize=(15, 10))
    for i, col in enumerate(features):
        plt.subplot(2, 2, i + 1)
        x = df.sample(1000)
        sb.scatterplot(x=col, y=target, data=x)
    plt.tight_layout()
    plt.show()

def distribution_plots(df, features):
    plt.subplots(figsize=(15, 10))
    for i, col in enumerate(features):
        plt.subplot(2, 3, i + 1)
        sb.histplot(df[col], kde=True)
    plt.tight_layout()
    plt.show()
