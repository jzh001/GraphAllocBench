import pandas as pd
import plotnine as p9
import numpy as np
from pathlib import Path

def plot_summary_stats(pcpl_csv_path,
                       pdmorl_csv_path,
                       data_dir = "data"):

    # Load PCPL data
    df_pcpl = pd.read_csv(pcpl_csv_path)
    df_pcpl['Problem'] = df_pcpl['env_name'].apply(lambda s: s.split('_')[1])
    df_pcpl['Architecture'] = 'PCPL'
    # Select and rename columns
    df_pcpl = df_pcpl[['Problem', 'normalized_hv', 'non_dominated', 'ordering', 'seed', 'Architecture']]
    
    # Load PDMORL data
    df_pdmorl = pd.read_csv(pdmorl_csv_path)
    df_pdmorl['Problem'] = df_pdmorl['problem'].apply(lambda s: s.split('_')[1])
    df_pdmorl['Architecture'] = 'PD-MORL'
    # Rename columns to match PCPL
    df_pdmorl = df_pdmorl.rename(columns={
        'percent_non_dominated': 'non_dominated',
        'ordering_score': 'ordering'
    })
    df_pdmorl = df_pdmorl[['Problem', 'normalized_hv', 'non_dominated', 'ordering', 'seed', 'Architecture']]

    # Combine
    df = pd.concat([df_pcpl, df_pdmorl], ignore_index=True)

    # Melt
    plot_df = ( 
        df
        .set_index(['Problem','seed', 'Architecture'])
        .rename_axis('Metric',axis=1)
        .stack()
    ).rename('Score').reset_index()

    metric_map = {
        'normalized_hv': 'HV Ratio',
        'non_dominated': '% Non-Dominated',
        'ordering': 'Ordering Score',
    }
    plot_df['Metric'] = pd.Categorical(
        values=plot_df['Metric'].map(metric_map),
        categories=['HV Ratio', '% Non-Dominated', 'Ordering Score'],
        ordered=True
    )

    p = (
        p9.ggplot(data=plot_df, mapping=p9.aes(x='Problem', y='Score', color='Architecture', fill='Architecture'))
        + p9.stat_summary(
            geom='point', 
            fun_y=np.mean, 
            size=10,
            shape='_',
            stroke=1.5,
            position=p9.position_dodge(width=0.6),
            mapping=p9.aes(group='Architecture', color='Architecture')
        )
        + p9.geom_point(
            position=p9.position_dodge(width=0.6),
            size=2.5,
            alpha=0.4,
            stroke=0
        )
        + p9.facet_grid('Metric~.')
        + p9.theme_classic()
        + p9.theme(
            panel_grid_major_y=p9.element_line(color='#e5e5e5', size=0.5),
            panel_grid_major_x=p9.element_line(color='#e5e5e5', size=0.5),
            legend_position='top',
            strip_background=p9.element_rect(fill='#f0f0f0', color='none'),
        )
        + p9.scale_fill_brewer(type='qual', palette='Dark2')
        + p9.scale_color_brewer(type='qual', palette='Dark2')
    )

    # Save the plot as PDF and PNG
    if isinstance(data_dir, str):
        output_dir = Path(data_dir)
    else:
        output_dir = data_dir
        
    output_dir.mkdir(parents=True, exist_ok=True)
    p.save(filename=str(output_dir / f"summary.pdf"), format="pdf")
    p.save(filename=str(output_dir / f"summary.png"), format="png")

    return p

if __name__ == "__main__":
    plot_summary_stats(pcpl_csv_path="data/pcpl_stats.csv",
                       pdmorl_csv_path="data/pdmorl_stats.csv"
                       )