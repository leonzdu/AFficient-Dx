import ast
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from afficient_dx.metrics import apply_calibrator, bootstrap_ci, reliability_table, scores

def save_curves(out, prefix, y, p, threshold, plots=False, bins=10, ranking_scores=None):
    ranking = p if ranking_scores is None else ranking_scores
    score_column = 'threshold_logit' if ranking_scores is not None else 'threshold_probability'
    fpr, tpr, roc_thresholds = roc_curve(y, ranking, drop_intermediate=False)
    precision, recall, pr_thresholds = precision_recall_curve(y, ranking)
    pd.DataFrame({'fpr': fpr, 'tpr': tpr, score_column: roc_thresholds}).to_csv(out / f'{prefix}_roc.csv', index=False)
    pd.DataFrame({'precision': precision, 'recall': recall, score_column: np.r_[pr_thresholds, np.nan]}).to_csv(out / f'{prefix}_pr.csv', index=False)
    reliability = pd.DataFrame(reliability_table(y, p, bins))
    reliability.to_csv(out / f'{prefix}_reliability.csv', index=False)
    matrix = confusion_matrix(y, p >= threshold, labels=[0, 1])
    pd.DataFrame(matrix, index=['actual_non_AF', 'actual_AF'], columns=['predicted_non_AF', 'predicted_AF']).to_csv(out / f'{prefix}_confusion.csv')
    if not plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes[0, 0].plot(fpr, tpr)
    axes[0, 0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0, 0].set(xlabel='False positive rate', ylabel='Sensitivity', title='ROC')
    axes[0, 1].step(recall, precision, where='post')
    axes[0, 1].axhline(y.mean(), linestyle='--', color='gray')
    axes[0, 1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
    nonempty = reliability[reliability.n > 0]
    axes[1, 0].plot(nonempty.mean_probability, nonempty.af_fraction, 'o-')
    axes[1, 0].plot([0, 1], [0, 1], '--', color='gray')
    axes[1, 0].set(xlabel='Mean predicted probability', ylabel='Observed AF fraction', title='Calibration', xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, str(matrix[i, j]), ha='center', va='center')
    axes[1, 1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['non-AF', 'AF'], yticklabels=['non-AF', 'AF'], xlabel='Predicted', ylabel='Actual', title=f'Confusion matrix (threshold={threshold:.3g})')
    fig.tight_layout()
    fig.savefig(out / f'{prefix}_diagnostics.png', dpi=180)
    plt.close(fig)

def prediction_frame(df, y, logits, p, threshold):
    frame = df.reset_index().copy()
    frame['y'] = y.astype(int)
    frame['logit'] = logits
    frame['raw_probability'] = expit(logits)
    frame['probability'] = p
    frame['prediction'] = (p >= threshold).astype(int)
    frame['threshold'] = threshold
    frame['outcome'] = np.where(y == 1, np.where(p >= threshold, 'TP', 'FN'), np.where(p >= threshold, 'FP', 'TN'))
    frame['error_confidence'] = np.where(y == 1, 1 - p, p)
    return frame

def failure_analysis(out, prefix, frame, bins=10):
    errors = frame[frame.outcome.isin(['FP', 'FN'])].sort_values('error_confidence', ascending=False)
    errors.to_csv(out / f'{prefix}_failure_cases.csv', index=False)
    groups = []
    if 'age' in frame:
        age = pd.to_numeric(frame.age, errors='coerce')
        frame = frame.assign(age_group=pd.cut(age, [-np.inf, 40, 65, np.inf], right=False, labels=['under_40', '40_to_64', '65_plus']).astype(str))
    for column in ['sex', 'age_group', 'static_noise', 'burst_noise', 'baseline_drift', 'electrodes_problems', 'extra_beats', 'pacemaker']:
        if column not in frame:
            continue
        values = frame[column].fillna('missing').astype(str)
        if column not in ['sex', 'age_group']:
            values = values.map(lambda v: 'unannotated' if v in ['missing', '', '0', '0.0'] else 'annotated')
        for group in sorted(values.unique()):
            mask = values == group
            group_scores = scores(frame.loc[mask, 'y'].to_numpy(), frame.loc[mask, 'probability'].to_numpy(), frame.threshold.iloc[0], bins, frame.loc[mask, 'logit'].to_numpy())
            groups.append({'field': column, 'group': group, 'exploratory': True, **group_scores})
    code_sets = frame.scp_codes.map(lambda s: set(ast.literal_eval(s)))
    for code in sorted(set().union(*code_sets)):
        mask = code_sets.map(lambda values: code in values)
        if int(mask.sum()) >= 20:
            groups.append({'field': 'overlapping_scp_code', 'group': code, 'exploratory': True, **scores(frame.loc[mask, 'y'].to_numpy(), frame.loc[mask, 'probability'].to_numpy(), frame.threshold.iloc[0], bins, frame.loc[mask, 'logit'].to_numpy())})
    pd.DataFrame(groups).to_csv(out / f'{prefix}_subgroups.csv', index=False)
    return {'false_positives': int((frame.outcome == 'FP').sum()), 'false_negatives': int((frame.outcome == 'FN').sum()), 'scope': 'descriptive associations and ranked errors, not adjudicated clinical causes'}

def evaluate_predictions(out, prefix, df, y, z, calibration, threshold, args, bootstrap=False):
    p = apply_calibrator(z, calibration)
    frame = prediction_frame(df, y, z, p, threshold)
    frame.to_csv(out / f'{prefix}_predictions.csv', index=False)
    result = {'primary': scores(y, p, threshold, args.calibration_bins, z), 'raw_fixed_0_5': scores(y, expit(z), 0.5, args.calibration_bins, z), 'calibrated_fixed_0_5': scores(y, p, 0.5, args.calibration_bins, z)}
    save_curves(out, prefix, y, p, threshold, args.plots, args.calibration_bins, z)
    pd.DataFrame(reliability_table(y, expit(z), args.calibration_bins)).to_csv(out / f'{prefix}_raw_reliability.csv', index=False)
    if bootstrap:
        result['bootstrap_95_ci'] = bootstrap_ci(y, p, df.patient_id.to_numpy(), args.bootstrap, args.seed + 1000, threshold, args.calibration_bins, z)
        result['failure_analysis'] = failure_analysis(out, prefix, frame, args.calibration_bins)
    return result
