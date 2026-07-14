import sys, numpy as np
sys.path.insert(0, '.')
import wfdb
from app.core.signal_processing import process_signal, PRE_SAMPLES_360, POST_SAMPLES_360
from app.core.features import extract_features, N_FEATURES, FEATURE_NAMES

rec = wfdb.rdrecord('100', pn_dir='mitdb', sampto=10800)
ann = wfdb.rdann('100', 'atr', pn_dir='mitdb', sampto=10800)
signal = rec.p_signal[:, 0].astype('float32')
fs = int(rec.fs)

proc = process_signal(signal, fs, ann_rpeaks=ann.sample)
n_beats = len(proc['beats'])
win_len = len(proc['beats'][0])
expected = int(100*fs/360) + int(200*fs/360)
print('Beats:', n_beats, '| Window length:', win_len, '(expected ~', expected, ')')
print('beat_starts present:', len(proc['beat_starts']) == n_beats)

val = proc['rpeak_validation']
print('R-peak sensitivity: %.4f  PPV: %.4f  F1: %.4f' % (val['sensitivity'], val['ppv'], val['f1']))

pre = int(100 * fs / 360)
X = extract_features(proc['beats'], proc['rpeaks'], fs, pre_samples=pre)
print('Feature matrix:', X.shape, '(expected n_beats x %d)' % N_FEATURES)
print('Features[0:7]:', FEATURE_NAMES[:7])

print('curr_rr[1]: %.1f ms  prev_rr[1]: %.1f ms  ratio: %.3f  variability: %.1f ms' % (
    X[1,0], X[1,1], X[1,2], X[1,3]))
print('p_present[0]: %.1f  qrs_dur[0]: %.1f ms  st_elev[0]: %.4f' % (
    X[0, FEATURE_NAMES.index('p_present')],
    X[0, FEATURE_NAMES.index('qrs_duration_ms')],
    X[0, FEATURE_NAMES.index('st_elevation')]))
print('Pipeline v2: PASSED')
