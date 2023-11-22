import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def plot_3d_scatter(data, x_col,y_col,z_col,color_col, x_title=None, y_title=None, z_title=None):
    # Create a 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=data[x_col],
        y=data[y_col],
        z=data[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=data[color_col],
            colorscale='Viridis',
            opacity=0.8
        )
    ))

    if x_title is None: 
        x_title = x_col
    if y_title is None:
        y_title = y_col
    if z_title is None:
        z_title = z_col

    title = x_title + ', ' + y_title + ', ' + z_title

    # Set labels and title
    fig.update_layout(scene=dict(
        xaxis_title=x_title,
        yaxis_title=y_title,
        zaxis_title=z_title),
        title=title
    )

    # Display the plot
    fig.show()


def multiplot_bar(): 
    # Obtain coefficients
    coefs_001 = lasso_0_01.coef_
    coefs_01 = lasso_0_1.coef_
    coefs_05 = lasso_0_5.coef_

    # Create bar plots for coefficients
    labels = X.columns

    # Set common y-axis limits
    y_min = -4
    y_max = 4

    # Create subplots with shared y-axis
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

    # Linear Regression Coefficients
    axes[0].bar(labels, coefs_001, color='b', alpha=0.7)
    axes[0].set_title('Coefficients Alpha = 0.01')
    axes[0].set_xlabel('Features')
    axes[0].set_ylabel('Coefficient')
    axes[0].set_ylim([y_min, y_max])

    # Lasso Regression Coefficients
    axes[1].bar(labels, coefs_01, color='g', alpha=0.7)
    axes[1].set_title('Coefficients Alpha = 0.1')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Coefficient')
    axes[1].set_ylim([y_min, y_max])

    # Ridge Regression Coefficients
    axes[2].bar(labels, coefs_05, color='r', alpha=0.7)
    axes[2].set_title('Coefficients Alpha = 0.5')
    axes[2].set_xlabel('Features')
    axes[2].set_ylabel('Coefficient')
    axes[2].set_ylim([y_min, y_max])

    plt.tight_layout()
    plt.show()

def plot_heat_map(df):
    df_corr = df.corr()

    # Plot correlations
    # Remove upper triangle
    fig, ax = plt.subplots(figsize=(14,8))
    ax = sns.heatmap(df_corr, annot = True)