"""Build a local SHA256 snapshot; does not fetch sources or infer ZIP commits."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.name == 'stage2_manifest.json':
            continue
        files.append(dict(path=path.relative_to(ROOT).as_posix(),
                          bytes=path.stat().st_size, sha256=sha256(path)))
    raw = ROOT / 'references/amgc_source/AMGC-master/datasets/data/cora/Cora/raw'
    comparison = []
    for local in sorted((ROOT / 'data/planetoid').glob('ind.cora.*')):
        other = raw / local.name
        comparison.append(dict(name=local.name, official_copy_exists=other.is_file(),
                               identical=sha256(local) == sha256(other) if other.is_file() else None))
    query = json.loads((ROOT / 'references/amgc_commit.json').read_text(encoding='utf-8-sig'))
    manifest = dict(
        schema_version=1,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        root=str(ROOT),
        scope='Local snapshot, including scripts, reports, data, source archive, model states and saved predictions; excludes pycache and this manifest.',
        sources=[
            dict(title='Structural Deep Clustering Network',
                 paper_url='https://arxiv.org/pdf/2002.01633',
                 local_pdf='references/sdcn.pdf', pages=11,
                 code_url='https://github.com/bdy9527/SDCN',
                 code_version='Downloaded master files, not commit-pinned'),
            dict(title='On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs with Missing Node Features',
                 paper_url='https://arxiv.org/pdf/2111.12128', local_pdf='references/fp.pdf',
                 code_url='https://github.com/twitter-research/feature-propagation',
                 code_commit_for_fp_official_py='af2733589eab4023fca67f7e71a3b46ddbbea8cd'),
            dict(title='Attribute-Missing Graph Clustering Network',
                 doi='10.1609/aaai.v38i14.29464',
                 paper_url='https://ojs.aaai.org/index.php/AAAI/article/download/29464/30760',
                 local_pdf='references/amgc.pdf', pages=10,
                 code_url='https://github.com/WxTu/AMGC',
                 archive_url='https://codeload.github.com/WxTu/AMGC/zip/refs/heads/master',
                 queried_master_commit=query.get('sha'),
                 archive_commit_verified=False,
                 archive_identity='Use ZIP and individual file hashes; queried commit is not proof of archive revision'),
            dict(title='Cora Planetoid raw data',
                 url='https://github.com/kimiyoung/planetoid/tree/master/data',
                 local_directory='data/planetoid',
                 version='Downloaded raw files, not commit-pinned; see local SHA256')],
        corrections=[
            dict(record='download_manifest.json: SDCN download marked ok',
                 status='invalid as SDCN evidence',
                 reason='arXiv 2004.10417 is a different mathematics paper',
                 retained_as='references/rejected_arxiv_2004.10417.pdf',
                 replacement='references/sdcn.pdf from arXiv 2002.01633; validated title and 11 pages'),
            dict(record='download_manifest.json: amgc_openreview.html marked ok',
                 status='not paper content', reason='Cloudflare challenge page'),
            dict(record='AMGC OpenReview PDF 403 / SDCN correct PDF timeout / Cora ally timeout',
                 status='historical attempts retained',
                 resolution='Current AMGC PDF obtained from AAAI; correct SDCN PDF obtained; all 8 Cora raw files present. Current local hashes supersede historical path statuses.')],
        cora_raw_comparison_with_amgc=comparison,
        validation='sdcn_calibration_100_400/verification.json',
        experiments=dict(main_runs=10, masks_per_rate=1, model_seeds=[0, 1],
                         sdcn_status='adaptation, not official Cora reproduction',
                         amgc_status='paper and static source audit only; no training run'),
        files=files)
    destination = ROOT / 'stage2_manifest.json'
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(files=len(files), queried_amgc_commit=query.get('sha'),
                          cora_comparison=comparison), ensure_ascii=True))


if __name__ == '__main__':
    main()
