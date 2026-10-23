"""Compare two raw fp32 engine dumps, reporting numeric drift without a bit-exact bar."""
import argparse
import json
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('cpu')
ap.add_argument('gpu')
args = ap.parse_args()
a, b = np.fromfile(args.cpu, dtype='<f4'), np.fromfile(args.gpu, dtype='<f4')
if a.size != b.size or not a.size:
    raise SystemExit('dump sizes differ or are empty')
diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
print('RESULT ' + json.dumps({'bench':'gpu-engine-logits', 'words':int(a.size),
    'finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),
    'different_words':int(np.count_nonzero(a.view('u4') != b.view('u4'))),
    'max_abs':float(diff.max()), 'mean_abs':float(diff.mean()),
    'relative_max':float(diff.max() / max(float(np.max(np.abs(a))), 1e-30)),
    'argmax_identical':bool(np.argmax(a) == np.argmax(b))}))
