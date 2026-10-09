#!/usr/bin/env python3
"""Build a standalone heading-linked PCA explorer from the verified Sim2 cache."""
import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--template', type=Path)
    parser.add_argument('--image-base', default='https://raw.githubusercontent.com/fwromano/UCLIP/77dee02fe81b28ba2f2c33e46a9a905e4c7498b5/data/car_sim/')
    parser.add_argument('--labels-url', default='https://fwromano.github.io/datasets/sim2-headings/')
    args = parser.parse_args()
    cache = args.repo_root / 'data/car_sim/clip_embeddings'
    output = args.output or cache
    template = args.template or args.repo_root / 'src/uclip/viz/sim2_embedding_orbit.html'
    manifest = json.loads((cache / 'manifest.json').read_text())
    for name, metadata in manifest['artifacts'].items():
        if digest(cache / name) != metadata['sha256']:
            raise ValueError(f'Cache integrity failure: {name}')
    index = json.loads((cache / 'index.json').read_text())['images']
    vectors = np.load(cache / 'embeddings.npy', allow_pickle=False)
    if vectors.shape != (len(index), 512) or not np.isfinite(vectors).all():
        raise ValueError('Expected a complete 512-dimensional embedding cache')
    rows = [r for r in index if r['vehicle_type'] == 'jeep']
    selected_indices = [r['embedding_row'] for r in rows]
    matrix = vectors[selected_indices].astype(np.float64)
    # Reuse UCLIP's established centered-SVD PCA rather than fitting in the browser.
    sys.path.insert(0, str(args.repo_root))
    from scripts.clip_mcdo_pca import fit_pca
    mean, components, explained, ratio = fit_pca(matrix, n_components=3)
    coords = (matrix - mean) @ components.T
    np.testing.assert_allclose(components @ components.T, np.eye(3), atol=1e-12)
    assert len(rows) == 1404 and len(set(selected_indices)) == 1404
    colors = {'Indigo': '#9d86ff', 'Magenta': '#f56dbd', 'White': '#d7e5f5', 'Yellow': '#f8ca56'}
    references = {}
    for category in colors:
        candidates = [i for i, r in enumerate(rows) if r['category'] == category]
        references[category] = min(candidates, key=lambda i: (abs((rows[i]['relative_heading_deg'] + 180) % 360 - 180), rows[i]['embedding_row']))
    browser_rows, exported_rows = [], []
    for i, row in enumerate(rows):
        path = args.repo_root / 'data/car_sim' / row['image_path']
        if digest(path) != row['image_sha256']:
            raise ValueError(f"Image integrity failure: {row['image_path']}")
        with Image.open(path) as source:
            image = source.convert('RGB')
            image.thumbnail((384, 256))
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=83)
        record = {'embedding_row': row['embedding_row'], 'category': row['category'],
                  'heading': row['relative_heading_deg'], 'image_path': row['image_path'],
                  'image_sha256': row['image_sha256'], 'uncertainty': row['heading_uncertainty_deg'],
                  'coords': coords[i].tolist(),
                  'cos_to_front': float(matrix[i] @ matrix[references[row['category']]])}
        exported_rows.append(record)
        browser_rows.append(dict(record, preview='data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()))
    projection = {'schema': 'sim2.heading_pca.v1', 'fit': 'shared centered PCA over all 1404 normalized Jeep image embeddings',
                  'model': manifest['model'], 'embedding_manifest_sha256': digest(cache / 'manifest.json'),
                  'embedding_sha256': manifest['artifacts']['embeddings.npy']['sha256'],
                  'components': components.tolist(), 'mean': mean.tolist(),
                  'explained_variance': explained.tolist(), 'explained_variance_ratio': ratio.tolist(),
                  'references': {name: rows[i]['embedding_row'] for name, i in references.items()},
                  'images': exported_rows,
                  'path_convention': 'One representative per measured integer heading, lowest embedding_row wins; ascending heading. Solid links span <=5 degrees; larger gaps and the last-to-first seam are dashed guides. No synthetic features or loop constraint.'}
    payload = {'rows': browser_rows, 'colors': colors, 'ratio': ratio.tolist(),
               'references': projection['references'], 'model': manifest['model'],
               'imageBase': args.image_base, 'labelsUrl': args.labels_url,
               'cacheUrl': 'https://github.com/fwromano/UCLIP/tree/77dee02fe81b28ba2f2c33e46a9a905e4c7498b5/data/car_sim/clip_embeddings'}
    html = template.read_text().replace('__PLOTLY_JS__', get_plotlyjs())
    html = html.replace('__ORBIT_DATA__', json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'heading_pca.html').write_text(html)
    (output / 'heading_pca.json').write_text(json.dumps(projection, indent=2, allow_nan=False) + '\n')
    verification = {'schema': 'sim2.heading_pca_verification.v1', 'passed': True,
                    'images_and_vector_rows_verified': len(rows), 'projection': 'centered SVD, reused scripts.clip_mcdo_pca.fit_pca',
                    'orthonormal_basis_max_error': float(np.max(np.abs(components @ components.T - np.eye(3)))),
                    'three_component_variance_fraction': float(ratio.sum()),
                    'generator_sha256': digest(__file__), 'template_sha256': digest(template),
                    'pca_implementation_sha256': digest(args.repo_root / 'scripts/clip_mcdo_pca.py'),
                    'files': {name: {'sha256': digest(output / name), 'bytes': (output / name).stat().st_size}
                              for name in ('heading_pca.html', 'heading_pca.json')}}
    (output / 'heading_pca_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
