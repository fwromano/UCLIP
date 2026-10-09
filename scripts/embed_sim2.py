#!/usr/bin/env python3
"""Build or independently verify a hash-bound, deterministic Sim2 CLIP cache."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch
import transformers
from transformers import CLIPImageProcessor, CLIPModel

MODEL = 'openai/clip-vit-base-patch16'
REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def encode(model, processor, paths, device):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    inputs = processor(images=images, return_tensors='pt').to(device)
    with torch.inference_mode():
        raw = model.get_image_features(**inputs).float()
        norms = raw.norm(dim=-1, keepdim=True)
        if not torch.isfinite(raw).all() or not (norms > 0).all():
            raise ValueError('Encoder returned invalid image features')
        unit = raw / norms
    return unit.cpu().numpy().astype(np.float32), norms[:, 0].cpu().numpy().astype(np.float32)


def main():
    root = Path(__file__).resolve().parents[1] / 'data/car_sim'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=root)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-name', default=MODEL)
    parser.add_argument('--revision', help='Immutable Hugging Face commit; defaults to the pinned B/16 revision')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--verify', action='store_true', help='Check all cached rows and independently re-encode selected crops')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('Batch size and threads must be positive')
    revision = args.revision or (REVISION if args.model_name == MODEL else None)
    if not revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        parser.error('An immutable 40-character model revision is required')
    output = args.output or args.data_root / 'clip_embeddings'
    labels_path = args.data_root / 'heading_labels/heading_labels.json'
    labels = read_json(labels_path)
    rows = labels['images']
    if labels['schema'] != 'sim2.relative_heading_labels.v1' or not rows:
        raise ValueError('Expected the complete Sim2 relative heading manifest')
    if len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image paths')
    for row in rows:
        path = (args.data_root / row['image_path']).resolve()
        if not path.is_relative_to(args.data_root.resolve()) or sha256(path) != row['image_sha256']:
            raise ValueError(f"Image path/hash mismatch: {row['image_path']}")
    if not args.verify and output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; use a new output directory')
    if args.verify:
        manifest = read_json(output / 'manifest.json')
        if (manifest['model']['name'], manifest['model']['revision']) != (args.model_name, revision):
            raise ValueError('Requested model does not match the cached model')
        if manifest['source_labels']['sha256'] != sha256(labels_path):
            raise ValueError('Heading manifest changed')
        for name, metadata in manifest['artifacts'].items():
            if sha256(output / name) != metadata['sha256']:
                raise ValueError(f'Artifact hash mismatch: {name}')
        index = read_json(output / 'index.json')
        if len(index['images']) != len(rows):
            raise ValueError('Incomplete embedding index')
        for number, (source, cached) in enumerate(zip(rows, index['images'])):
            if any(cached.get(key) != value for key, value in source.items()) or cached['embedding_row'] != number:
                raise ValueError(f'Incorrect row association: {number}')
        vectors = np.load(output / 'embeddings.npy', allow_pickle=False)
        raw_norms = np.load(output / 'raw_norms.npy', allow_pickle=False)
        if vectors.shape != (len(rows), manifest['embedding']['dimensions']) or vectors.dtype != np.float32:
            raise ValueError('Invalid vector shape/dtype')
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6):
            raise ValueError('Invalid normalized vectors')
        if raw_norms.shape != (len(rows),) or not np.isfinite(raw_norms).all() or not (raw_norms > 0).all():
            raise ValueError('Invalid raw projection norms')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    options = {'revision': revision, 'cache_dir': args.cache_dir}
    processor = CLIPImageProcessor.from_pretrained(args.model_name, **options)
    model = CLIPModel.from_pretrained(args.model_name, **options).to(args.device).eval()
    started = time.monotonic()
    if args.verify:
        # Different batch composition from generation; four cardinal views per category.
        selected = sorted({min((i for i, r in enumerate(rows) if r['category'] == category),
                               key=lambda i: abs((rows[i]['relative_heading_deg'] - angle + 180) % 360 - 180))
                           for category in {r['category'] for r in rows} for angle in (0, 90, 180, 270)})
        maximum = 0.0
        for number in selected:
            recomputed, norm = encode(model, processor, [args.data_root / rows[number]['image_path']], args.device)
            maximum = max(maximum, float(np.max(np.abs(recomputed[0] - vectors[number]))))
            np.testing.assert_allclose(recomputed[0], vectors[number], atol=2e-6, rtol=1e-5)
            np.testing.assert_allclose(norm[0], raw_norms[number], atol=1e-4, rtol=1e-5)
        result = {'schema': 'sim2.clip_cache_verification.v1', 'passed': True,
                  'rows_verified': len(rows), 'image_hashes_verified': len(rows),
                  'independent_single_image_encodes': len(selected), 'selected_rows': selected,
                  'max_absolute_embedding_difference': maximum,
                  'manifest_sha256': sha256(output / 'manifest.json'),
                  'verification_device': args.device, 'elapsed_sec': time.monotonic() - started}
        write_json(output / 'verification.json', result)
        print(json.dumps(result), flush=True)
        return
    vectors, raw_norms = [], []
    for begin in range(0, len(rows), args.batch_size):
        batch = rows[begin:begin + args.batch_size]
        unit, norm = encode(model, processor, [args.data_root / r['image_path'] for r in batch], args.device)
        vectors.append(unit); raw_norms.append(norm)
        print(f'{begin + len(batch)}/{len(rows)} images; {time.monotonic() - started:.1f}s', flush=True)
    vectors, raw_norms = np.concatenate(vectors), np.concatenate(raw_norms)
    if vectors.shape != (len(rows), model.config.projection_dim):
        raise ValueError('Incomplete feature matrix')
    output.mkdir(parents=True)
    np.save(output / 'embeddings.npy', vectors, allow_pickle=False)
    np.save(output / 'raw_norms.npy', raw_norms, allow_pickle=False)
    write_json(output / 'index.json', {'schema': 'sim2.clip_embedding_index.v1',
               'vector_file': 'embeddings.npy', 'raw_norm_file': 'raw_norms.npy',
               'images': [dict(row, embedding_row=i) for i, row in enumerate(rows)]})
    manifest = {'schema': 'sim2.clip_embedding_cache.v1',
                'model': {'name': args.model_name, 'revision': revision},
                'embedding': {'dimensions': int(vectors.shape[1]), 'dtype': 'float32',
                              'normalization': 'L2 unit length', 'feature': 'CLIP projected image features',
                              'raw_reconstruction': 'embeddings[row] * raw_norms[row]',
                              'inference': 'eval, no dropout sampling, torch.inference_mode, float32'},
                'preprocessing': {'implementation': 'transformers.CLIPImageProcessor',
                                  'input_conversion': 'PIL RGB', 'configuration': processor.to_dict()},
                'source_labels': {'path': 'heading_labels/heading_labels.json', 'sha256': sha256(labels_path)},
                'generation': {'device': args.device, 'batch_size': args.batch_size, 'threads': args.threads,
                               'seed': 0, 'deterministic_algorithms': True, 'tf32': False,
                               'elapsed_sec': time.monotonic() - started},
                'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                             'transformers': transformers.__version__, 'numpy': np.__version__, 'Pillow': PIL.__version__},
                'rows': len(rows), 'artifacts': {}}
    for name in ('embeddings.npy', 'raw_norms.npy', 'index.json'):
        manifest['artifacts'][name] = {'sha256': sha256(output / name), 'bytes': (output / name).stat().st_size}
    write_json(output / 'manifest.json', manifest)
    print(f'Cache saved: {output}; shape={vectors.shape}', flush=True)


if __name__ == '__main__':
    main()
