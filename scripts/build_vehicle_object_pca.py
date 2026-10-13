#!/usr/bin/env python3
"""Render a local, single-series PCA using every indexed cropped CLIP view."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vehicle-object-pca-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--crops', type=Path, required=True)
    parser.add_argument('--series', default='truck/yellow')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index_path = args.crops / 'index.json'
    feature_path = args.data / 'cropped_embeddings.npy'
    index = json.loads(index_path.read_text())
    assert index['images'] == json.loads((args.data / 'crop_index.json').read_text())['images']
    evaluation = json.loads((args.data / 'evaluation.json').read_text())
    holdout = json.loads((args.data / 'elevation_holdout.json').read_text())
    assert sha(index_path) == evaluation['source_crop_index_sha256']
    assert sha(feature_path) == holdout['feature_files']['cropped_embeddings.npy']
    features = np.load(feature_path, allow_pickle=False)
    rows = sorted([r for r in index['images'] if r['series'] == args.series],
                  key=lambda r: (r['elevation_deg'], r['source_azimuth_deg']))
    elevations = sorted({r['elevation_deg'] for r in rows})
    expected = {(e, a) for e in [-5, 0, 18, 36, 54, 72, 90] for a in range(0, 360, 15)}
    assert len(rows) == 168 and {(r['elevation_deg'], r['source_azimuth_deg']) for r in rows} == expected
    assert len({r['source_sha256'] for r in rows}) == len(rows)
    vectors = features[[r['embedding_row'] for r in rows]].astype(np.float64)
    assert vectors.shape == (168, 512) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6)
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    basis = components[:3].copy()
    for component in basis:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    coords = centered @ basis.T
    ratios = singular[:3] ** 2 / np.sum(singular ** 2)
    assert np.allclose(np.sum(coords ** 2, axis=0) / np.sum(centered ** 2), ratios)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    palette = plt.get_cmap('viridis')(np.linspace(.07, .95, len(elevations)))
    fig = plt.figure(figsize=(14, 11), layout='constrained')
    axes = [fig.add_subplot(2, 2, 1, projection='3d')]
    axes.extend(fig.add_subplot(2, 2, n) for n in (2, 3, 4))
    pairs = [(0, 1), (0, 2), (1, 2)]
    traces = []
    thumbs = []
    for r in rows:
        path = args.crops / r['crop_path']
        assert sha(path) == r['crop_sha256']
        with Image.open(path) as image:
            image.thumbnail((240, 160), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
        thumbs.append('data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())
    for elevation, color in zip(elevations, palette):
        color = tuple(color)
        indices = [i for i, r in enumerate(rows) if r['elevation_deg'] == elevation]
        points = coords[indices]
        closed = np.vstack([points, points[0]])
        axes[0].plot(*closed.T, color=color, alpha=.65, lw=1)
        axes[0].scatter(*points.T, color=color, s=28, depthshade=False, label=f'{elevation:g}°')
        for ax, (x, y) in zip(axes[1:], pairs):
            ax.plot(closed[:, x], closed[:, y], color=color, alpha=.65, lw=.9)
            ax.scatter(points[:, x], points[:, y], color=color, s=25)
        rgb = 'rgb(' + ','.join(str(int(c * 255)) for c in color[:3]) + ')'
        custom = [[i, rows[i]['source_azimuth_deg'], rows[i]['relative_heading_deg'], elevation] for i in indices]
        traces.append(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                  mode='markers', marker={'size': 5, 'color': rgb},
                                  name=f'{elevation:g}° elevation', customdata=custom,
                                  hovertemplate='Source azimuth %{customdata[1]}°<br>Relative heading %{customdata[2]}°<br>Elevation %{customdata[3]}°<extra></extra>'))
        traces.append(go.Scatter3d(x=closed[:, 0], y=closed[:, 1], z=closed[:, 2], mode='lines',
                                  line={'color': rgb, 'width': 2}, showlegend=False, hoverinfo='skip'))
    labels = [f'PC{i + 1} · {ratio * 100:.1f}% variance' for i, ratio in enumerate(ratios)]
    axes[0].set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2], title='All 168 views in three dimensions')
    axes[0].view_init(elev=24, azim=-52)
    axes[0].legend(title='Elevation', loc='upper left', fontsize=9)
    for ax, (x, y) in zip(axes[1:], pairs):
        ax.set(xlabel=labels[x], ylabel=labels[y], title=f'PC{x + 1} / PC{y + 1}')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
    fig.suptitle(f'{args.series} · one object, every scanned view\n24 azimuths × 7 elevations · cropped CLIP B/16 · dedicated PCA ({ratios.sum() * 100:.1f}% in 3 PCs)', fontsize=17)
    fig.supxlabel('Lines connect adjacent source azimuths within an elevation. At 90°, azimuth changes image orientation.\nPCA is fitted only to this object’s 168 views; it is separate from the combined Sim2/scan basis.', fontsize=10)
    fig.savefig(args.output / 'all-views.png', dpi=180)
    fig.savefig(args.output / 'all-views.svg')
    svg_path = args.output / 'all-views.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    plt.close(fig)
    chart = go.Figure(traces)
    chart.update_layout(height=690, margin={'l': 0, 'r': 0, 'b': 0, 't': 15},
                        scene={'xaxis_title': labels[0], 'yaxis_title': labels[1], 'zaxis_title': labels[2],
                               'aspectmode': 'data'}, legend={'title': {'text': 'Elevation'}})
    plot = pio.to_html(chart, full_html=False, include_plotlyjs=True, div_id='object-pca',
                       config={'responsive': True, 'displaylogo': False})
    summary = f'168 cropped views · 24 azimuths × 7 elevations · CLIP B/16 · first three PCs retain {ratios.sum()*100:.2f}% of variance'
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>One object — all views PCA</title><style>body{font:16px system-ui;margin:24px auto;max-width:1400px;padding:0 20px;color:#182b3b}h1{margin-bottom:8px}p{line-height:1.5}#preview{display:flex;gap:24px;align-items:center;background:#f2f5f7;padding:15px;border-radius:8px}#view{width:240px;object-fit:contain;image-rendering:auto}a{color:#1656a6}</style>'''
    html += f'<h1>{args.series} — every scanned view</h1><p>{summary}. PCA is fitted only to this object. Drag to rotate; hover over a point to inspect its crop. Click elevation labels to toggle them.</p>'
    html += plot + '<div id="preview"><img id="view" alt="Selected source crop"><div id="view-label"></div></div><p>Lines follow adjacent source azimuths within each elevation. The 90° ring records image rotation at nadir. Relative heading uses the reviewed front/rear convention. The basis differs from the combined Sim2/scan explorer. This is synthetic appearance evidence.</p><p><a href="all-views.png">PNG plot</a> · <a href="all-views.svg">SVG plot</a> · <a href="pca.json">Coordinates and basis</a></p>'
    records = [dict(r, coords=coords[i].tolist()) for i, r in enumerate(rows)]
    html += '<script>const views=' + json.dumps(thumbs) + ';const rows=' + json.dumps(records) + ';function show(i){const r=rows[i];document.getElementById("view").src=views[i];document.getElementById("view-label").textContent=`Source azimuth ${r.source_azimuth_deg}° · relative heading ${r.relative_heading_deg}° · elevation ${r.elevation_deg}°`;window.selectedObjectView=i;}document.getElementById("object-pca").on("plotly_hover",e=>{const d=e.points[0].customdata;if(d)show(d[0]);});show(0);</script></html>'
    (args.output / 'index.html').write_text(html)
    metadata = {'schema': 'uclip.single-object-pca.v1', 'series': args.series, 'rows': records,
                'view_count': len(rows), 'elevations_deg': elevations, 'azimuth_step_deg': 15,
                'model': evaluation['model'], 'processor': evaluation['processor'], 'framing': 'reviewed padded crops',
                'basis_scope': 'only the 168 selected object views; centered float64 SVD; largest loading positive',
                'mean': mean.tolist(), 'components': basis.tolist(), 'variance_ratio': ratios.tolist(),
                'source_crop_index_sha256': sha(index_path), 'source_features_sha256': sha(feature_path),
                'source_archive_sha256': index['source_archive_sha256'], 'generator_sha256': sha(Path(__file__)),
                'limits': ['Viewpoint and crop variation are coupled.', 'Nadir azimuth changes image orientation.',
                           'PCA is a lossy visualization, not a detection or identity accuracy metric.']}
    (args.output / 'pca.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'series': args.series, 'views': len(rows), 'variance_ratio': ratios.tolist(), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
