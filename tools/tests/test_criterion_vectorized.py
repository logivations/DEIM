"""
A/B check for the vectorized criterion internals (exp4-criterion-desync).

The old (pre-vectorization) implementations of the four refactored pieces are
kept in this file verbatim as references. The test compares them against the
new code on randomized synthetic data, both function-by-function and through a
full DEIMCriterion.forward() with aux/pre/enc groups. Runs on CPU:

    python3 tools/tests/test_criterion_vectorized.py

Note on _get_go_indices ties: when several (row, col) pairs of the same row
share the maximal count, the old dict loop depended on an UNSTABLE argsort, so
its pick was arbitrary. The new code breaks ties deterministically. The unit
check therefore asserts set-of-rows equality and max-count validity of every
pick, and exact col equality only for tie-free rows. The integration check
pins both criteria to the same go-indices so everything else must match
exactly.
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.deim.utils import filter_suppress_source_targets
from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iof

NUM_CLASSES = 10
SUPPRESS_CLASSES = {8: [2, 3], 9: [4]}
IGNORE_TAGS = {'pallets_annotated': [2, 3, 4]}


# ---------------------------------------------------------------- references
def old_filter_suppress_source_targets(targets, suppress_source_ids):
    if not suppress_source_ids:
        empty = [{k: v[:0] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                  for k, v in t.items()} for t in targets]
        return targets, empty
    main, sources = [], []
    for t in targets:
        mask = torch.tensor(
            [lab.item() not in suppress_source_ids for lab in t['labels']],
            dtype=torch.bool
        )
        source_mask = ~mask
        mt, st = {}, {}
        for k, v in t.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == mask.shape[0]:
                mt[k] = v[mask]
                st[k] = v[source_mask]
            else:
                mt[k] = v
                st[k] = v
        main.append(mt)
        sources.append(st)
    return main, sources


def old_get_ignore_tag_mask(crit, targets):
    bs = len(targets)
    device = targets[0]['labels'].device
    mask = torch.zeros(bs, crit.num_classes, dtype=torch.bool, device=device)
    for tag_name, class_ids in crit.ignore_tags_resolved.items():
        valid_ids = [c for c in class_ids if c < crit.num_classes]
        if not valid_ids:
            continue
        cls_tensor = torch.tensor(valid_ids, device=device)
        for i, t in enumerate(targets):
            if tag_name in t and t[tag_name].item() == 0:
                mask[i, cls_tensor] = True
    return mask if mask.any() else None


def old_get_suppress_mask(crit, outputs, suppress_source_targets, indices):
    bs, num_queries, num_classes = outputs['pred_logits'].shape
    device = outputs['pred_logits'].device
    suppress = torch.zeros(bs, num_queries, num_classes, dtype=torch.bool, device=device)
    matched = torch.zeros(bs, num_queries, dtype=torch.bool, device=device)
    for i, (src_idx, _) in enumerate(indices):
        matched[i, src_idx] = True
    for i, st in enumerate(suppress_source_targets):
        if len(st['boxes']) == 0:
            continue
        pred_boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'][i])
        source_boxes = box_cxcywh_to_xyxy(st['boxes'])
        iof_matrix = box_iof(pred_boxes, source_boxes)
        max_iof = iof_matrix.max(dim=1)[0]
        overlapping = (~matched[i]) & (max_iof > 0.5)
        if not overlapping.any():
            continue
        suppress_cls = set()
        for j in range(len(st['boxes'])):
            label = st['labels'][j].item()
            if label in crit.suppress_classes:
                suppress_cls.update(crit.suppress_classes[label])
        for cls_id in suppress_cls:
            suppress[i, overlapping, cls_id] = True
    return suppress


def old_get_go_indices(indices, indices_aux_list):
    results = []
    for indices_aux in indices_aux_list:
        indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                   for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]
    for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
        unique, counts = torch.unique(ind, return_counts=True, dim=0)
        count_sort_indices = torch.argsort(counts, descending=True)
        unique_sorted = unique[count_sort_indices]
        column_to_row = {}
        for idx in unique_sorted:
            row_idx, col_idx = idx[0].item(), idx[1].item()
            if row_idx not in column_to_row:
                column_to_row[row_idx] = col_idx
        final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
        final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
        results.append((final_rows.long(), final_cols.long()))
    return results


class OldPathCriterion(DEIMCriterion):
    """DEIMCriterion with the pre-vectorization implementations plugged back in."""

    def _get_ignore_tag_mask(self, targets):
        return old_get_ignore_tag_mask(self, targets)

    def _prepare_suppress_cache(self, suppress_source_targets):
        pass

    def _get_suppress_mask(self, outputs, suppress_source_targets, indices):
        return old_get_suppress_mask(self, outputs, suppress_source_targets, indices)


# ------------------------------------------------------------ data synthesis
def make_targets(gen, bs=4):
    targets = []
    for i in range(bs):
        n = int(torch.randint(0, 8, (1,), generator=gen))
        labels = torch.randint(0, NUM_CLASSES, (n,), generator=gen)
        if i % 2 == 0 and n > 0:  # guarantee some suppress sources
            labels[0] = 8 + (i // 2) % 2
        cxcy = torch.rand(n, 2, generator=gen) * 0.6 + 0.2
        wh = torch.rand(n, 2, generator=gen) * 0.2 + 0.05
        t = {'labels': labels, 'boxes': torch.cat([cxcy, wh], dim=1),
             'image_id': torch.tensor([i])}
        if i % 3 != 0:  # some images miss the tag entirely
            t['pallets_annotated'] = torch.tensor(int(torch.randint(0, 2, (1,), generator=gen)))
        targets.append(t)
    return targets


def make_layer_outputs(gen, bs=4, nq=30):
    return {
        'pred_logits': torch.randn(bs, nq, NUM_CLASSES, generator=gen),
        'pred_boxes': torch.rand(bs, nq, 4, generator=gen) * 0.5 + 0.25,
    }


def make_criterion(cls):
    crit = cls(
        matcher=HungarianMatcher(weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                                 alpha=0.25, gamma=2.0),
        weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
        losses=['vfl', 'boxes'],
        alpha=0.75, gamma=2.0, num_classes=NUM_CLASSES,
    )
    crit.suppress_classes = copy.deepcopy(SUPPRESS_CLASSES)
    crit.ignore_tags_resolved = copy.deepcopy(IGNORE_TAGS)
    return crit


# ------------------------------------------------------------------- checks
def check_filter(gen):
    targets = make_targets(gen)
    ids = set(SUPPRESS_CLASSES.keys())
    old_main, old_src = old_filter_suppress_source_targets(targets, ids)
    new_main, new_src = filter_suppress_source_targets(targets, ids)
    for om, nm in list(zip(old_main, new_main)) + list(zip(old_src, new_src)):
        assert om.keys() == nm.keys()
        for k in om:
            if isinstance(om[k], torch.Tensor):
                assert torch.equal(om[k], nm[k]), f'filter mismatch at {k}'
    print('OK: filter_suppress_source_targets')


def check_ignore_mask(gen):
    crit = make_criterion(DEIMCriterion)
    for _ in range(20):
        targets = make_targets(gen)
        old = old_get_ignore_tag_mask(crit, targets)
        new = crit._get_ignore_tag_mask(targets)
        if old is None:
            assert not new.any(), 'old=None but new mask not empty'
        else:
            assert torch.equal(old, new), 'ignore-tag mask mismatch'
    print('OK: _get_ignore_tag_mask')


def rand_indices(gen, bs, nq, targets):
    out = []
    for t in targets:
        n = min(len(t['labels']), nq)
        src = torch.randperm(nq, generator=gen)[:n].sort().values
        tgt = torch.randperm(max(n, 1), generator=gen)[:n]
        out.append((src, tgt))
    return out


def check_suppress_mask(gen):
    crit = make_criterion(DEIMCriterion)
    for _ in range(20):
        targets = make_targets(gen)
        _, src_targets = filter_suppress_source_targets(targets, set(SUPPRESS_CLASSES.keys()))
        outputs = make_layer_outputs(gen)
        indices = rand_indices(gen, len(targets), 30, targets)
        old = old_get_suppress_mask(crit, outputs, src_targets, indices)
        crit._prepare_suppress_cache(src_targets)
        new = crit._get_suppress_mask(outputs, src_targets, indices)
        assert torch.equal(old, new), 'suppress mask mismatch'
    print('OK: _get_suppress_mask')


def check_go_indices(gen):
    crit = make_criterion(DEIMCriterion)
    for _ in range(20):
        bs, nq, nt = 3, 20, 6
        def rand_layer():
            return [(torch.randint(0, nq, (nt,), generator=gen),
                     torch.randint(0, nt, (nt,), generator=gen)) for _ in range(bs)]
        indices, aux = rand_layer(), [rand_layer() for _ in range(3)]
        old = old_get_go_indices(indices, aux)
        new = crit._get_go_indices(indices, aux)
        for b, ((orows, ocols), (nrows, ncols)) in enumerate(zip(old, new)):
            # counts of every (row, col) pair, per batch element
            pairs = {}
            layers = [indices[b]] + [a[b] for a in aux]
            rows_cat = torch.cat([l[0] for l in layers]).tolist()
            cols_cat = torch.cat([l[1] for l in layers]).tolist()
            for r, c in zip(rows_cat, cols_cat):
                pairs[(r, c)] = pairs.get((r, c), 0) + 1
            assert set(orows.tolist()) == set(nrows.tolist()), 'row set mismatch'
            for r, c in zip(nrows.tolist(), ncols.tolist()):
                max_count = max(v for (rr, _), v in pairs.items() if rr == r)
                assert pairs[(r, c)] == max_count, \
                    f'new pick ({r},{c}) count {pairs[(r, c)]} < max {max_count}'
                tied = [cc for (rr, cc), v in pairs.items() if rr == r and v == max_count]
                if len(tied) == 1:
                    old_c = dict(zip(orows.tolist(), ocols.tolist()))[r]
                    assert c == old_c, f'tie-free row {r}: col {c} != old {old_c}'
    print('OK: _get_go_indices (set equality + max-count picks, exact on tie-free rows)')


def check_forward_integration(gen):
    torch.manual_seed(7)
    new_crit = make_criterion(DEIMCriterion)
    old_crit = make_criterion(OldPathCriterion)
    # pin both to the same go-indices so tie-breaks cannot differ
    old_crit._get_go_indices = new_crit._get_go_indices

    for trial in range(5):
        targets = make_targets(gen)
        bs, nq = len(targets), 30
        main = make_layer_outputs(gen, bs, nq)
        outputs = {
            **main,
            'aux_outputs': [make_layer_outputs(gen, bs, nq) for _ in range(2)],
            'pre_outputs': make_layer_outputs(gen, bs, nq),
            'enc_aux_outputs': [make_layer_outputs(gen, bs, nq)],
            'enc_meta': {'class_agnostic': False},
        }
        old_losses = old_crit(copy.deepcopy(outputs), copy.deepcopy(targets))
        new_losses = new_crit(copy.deepcopy(outputs), copy.deepcopy(targets))
        assert old_losses.keys() == new_losses.keys()
        for k in old_losses:
            assert torch.allclose(old_losses[k], new_losses[k], rtol=0, atol=1e-6), \
                f'trial {trial}, {k}: {old_losses[k].item()} vs {new_losses[k].item()}'
    print('OK: full criterion.forward (vfl+boxes over main/aux/pre/enc groups)')


def main():
    gen = torch.Generator().manual_seed(0)
    check_filter(gen)
    check_ignore_mask(gen)
    check_suppress_mask(gen)
    check_go_indices(gen)
    check_forward_integration(gen)
    print('\nAll criterion A/B checks passed.')


if __name__ == '__main__':
    main()
