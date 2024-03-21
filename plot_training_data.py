from plotnine import *
from IPython.display import display

def plot_data(interp_df, v_df, means, maxes, value_string):

    # First Plot
    plot = (ggplot(aes(y='Value', color='Model', group='ModelID'))
            + geom_line(data=v_df, alpha=0.5)
            + ylab(f'{value_string} (s)')
            + xlab('Training Experience (hours)')
            + theme_bw()
            + scale_x_continuous(breaks=range(int(min(v_df['Value'])),
                                              int(max(v_df['Value'])) + 1, 2))
            )
    #for model in v_df['Model'].unique():
    #plot += geom_line(aes(x='Time', y='Value'),
    #                  data=means,
    #                  group='Model')
    display(plot)

    # Second Plot
    plot = (ggplot(maxes, aes(x='Model',
                              y='MaxValue',
                              color='Model'))
            + geom_boxplot()
            + theme(legend_position='none')
            + geom_jitter(width=0.2, size=2, alpha=0.5)
            + ylab(f'Max {value_string} (s)')
            + theme(figure_size=(3, 5))
            + theme_bw())
    display(plot)
