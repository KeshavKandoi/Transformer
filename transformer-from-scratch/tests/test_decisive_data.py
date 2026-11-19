import random
from zipfile import ZipFile

import pytest

from scripts.decisive_data_audit import read_zip, stratified, stratum


def test_parallel_archive_checks_equal_counts_and_preserves_alignment(tmp_path):
    path = tmp_path / 'source.zip'
    with ZipFile(path,'w') as archive:
        archive.writestr('source.en','One.\nTwo.\n')
        archive.writestr('source.hi','एक।\nदो।\n')
    rows = read_zip(path)
    assert rows[1] == dict(index=2,source='Two.',target='दो।',origin='source')
    with ZipFile(path,'w') as archive:
        archive.writestr('source.en','One.\nTwo.\n')
        archive.writestr('source.hi','एक।\n')
    with pytest.raises(ValueError, match='unequal'):
        read_zip(path)


def test_stratified_respects_distinct_buckets_before_reusing_them():
    rows = [{'source':f'This is a sentence with number {i}.','target':'यह एक वाक्य है।'}
            for i in range(7)]
    rows.append({'source':'Please come home now.','target':'कृपया अब घर आ जाइए।'})
    selected = stratified(rows,2,random.Random(17))
    assert selected == stratified(rows,2,random.Random(17))
    assert len({stratum(row) for row in selected})==2
