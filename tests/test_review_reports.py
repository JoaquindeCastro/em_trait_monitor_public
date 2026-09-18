"""Checks for inline table rendering and compact activation inputs."""

from pathlib import Path

import torch

from src.reproduction.reports import table_frame


def test_table_frame_displays_values_without_a_latex_float(tmp_path):
    path = tmp_path / "table.tex"
    path.write_text(r"""\begin{table}
\caption{A caption}
\begin{tabular}{lcc}
\toprule
Model & FNR & FPR \\
\midrule
\textbf{RF} & $1.8\%$ & $2.0\%$ \\
\bottomrule
\end{tabular}
\end{table}
""")
    frame = table_frame(path)
    assert list(frame.columns) == ["Model", "FNR", "FPR"]
    assert frame.iloc[0].tolist() == ["RF", "1.8%", "2.0%"]


def test_compact_means_preserve_linear_projections_and_sae_on_mean():
    generator = torch.Generator().manual_seed(42)
    prompts = torch.randn(115, 16, generator=generator).to(torch.bfloat16)
    mean = prompts.float().mean(0, keepdim=True)
    projection = torch.randn(16, 7, generator=generator)
    encoder = torch.randn(16, 8, generator=generator)
    assert torch.equal(prompts.float().mean(0) @ projection, mean.mean(0) @ projection)
    assert torch.equal(torch.relu(prompts.float().mean(0) @ encoder),
                       torch.relu(mean.mean(0) @ encoder))
