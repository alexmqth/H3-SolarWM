"""CPU comparator controls from completed receipts; not candidate results."""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import report_density as report


def rejected(function):
    try:function()
    except ValueError:return
    raise AssertionError('Invalid comparison was accepted')


def main():
    before=report.read(report.NOISE/'step_00/probe.json')
    control=report.read(report.NOISE/'step_48/probe.json')
    rows=report.noise_rows([before,control,control])
    assert len(rows)==54 and all(r['shift12_raw_velocity_mse']==r['shift2_22_raw_velocity_mse'] for r in rows)
    bad=copy.deepcopy(control);bad['records'][0]['state_sha256']='deliberately_mismatched'
    rejected(lambda:report.noise_rows([before,control,bad]))
    bad=copy.deepcopy(control);bad['records'][0]['roles']['original']['prediction_sha256']='different_teacher'
    rejected(lambda:report.noise_rows([before,control,bad]))
    bad=copy.deepcopy(control);bad['records'].pop()
    rejected(lambda:report.noise_rows([before,control,bad]))
    base=report.read(report.CONTROL/'geometry/step_48_step00_generated/probe.json')
    identity=copy.deepcopy(base)
    # Explicit artificial identity fixture with the new bookkeeping source
    # declaration. No fixture is written as a GPU result or report.
    identity['source_sha256']['probe_real_geometry.py']=report.sha(report.BASE/'probe_real_geometry.py')
    rows=report.geometry_rows(base,identity)
    assert len(rows)==18 and all(r['shift12_delta_cosine']==r['shift2_22_delta_cosine'] for r in rows)
    bad=copy.deepcopy(identity);bad['records'][0]['pair_sha256']='wrong_action_pair'
    rejected(lambda:report.geometry_rows(base,bad))
    bad=copy.deepcopy(identity);bad['cache_audits'][0]['read_only']=False
    rejected(lambda:report.geometry_rows(base,bad))
    bad=copy.deepcopy(identity);bad['source_sha256']['probe_real_geometry.py']='unaudited_source'
    rejected(lambda:report.geometry_rows(base,bad))
    out=dict(status='passed',at=datetime.now().astimezone().isoformat(),checks=8,
        scope='CPU comparator identity and rejection fixtures only; not candidate model outputs',
        checks_detail=['noise identity','state hash mismatch rejected','teacher full hash mismatch rejected',
            'missing noise state rejected','geometry identity with audited path-only adaptation',
            'wrong action pair rejected','mutating KV rejected','unaudited source rejected'],
        report_source_sha256=report.sha(Path(report.__file__)))
    (report.BASE/'report_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
