"""Read-only release integrity and experiment catalogue inspector."""
import argparse, ast, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def verify():
    manifest=read(ROOT/"provenance/release_manifest.json")
    for retired in manifest["excluded"]:
        assert not (ROOT/retired).exists(), ("retired file present", retired)
    for row in manifest["files"]:
        p=ROOT/row["path"]
        assert p.is_file(),row["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row["published_sha256"],row["path"]
        if p.suffix==".json": read(p)
        if p.suffix==".py": ast.parse(p.read_text(encoding="utf-8-sig"),filename=str(p))
    for row in read(ROOT/"provenance/experiment_catalog.json")["experiments"]:
        for key in ("entry", "summary", "status"):
            assert (ROOT/row[key]).is_file(),(row["id"],key)
        assert read(ROOT/row["status"])["status"]=="complete",row["id"]
    bindings=read(ROOT/"provenance/manuscript_bindings_2026_09_20.json")
    numeric=read(ROOT/"provenance/manuscript_numeric_verification_2026_09_20.json")
    snapshot=read(ROOT/"provenance/manuscript_snapshot.json")
    assert bindings["manuscript"]==snapshot["files"]==numeric["manuscript"]["files"]
    assert not bindings["adaptation_integrated"] and not snapshot["adaptation_integrated"]
    assert [r["index"] for r in bindings["tables"]]==list(range(1,22))
    for row, original in zip(bindings["tables"],numeric["table_inventory"]):
        assert all(row[key]==original[key] for key in original)
        for path in row["evidence"]+row["implementation"]:
            assert (ROOT/path).is_file(),path
    assert hashlib.sha256((ROOT/bindings["figure_script"]).read_bytes()).hexdigest()==numeric["figure_script"]["sha256"]
    assert bindings["figure_artifacts"]==numeric["figure_artifacts"]
    print(json.dumps({"status":"pass","files_checked":len(manifest["files"]),"table_bindings":21,"figure_bindings":3,"scope":"Published integrity, syntax and manuscript bindings; no training or replay"}))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list",action="store_true");ap.add_argument("--verify",action="store_true");ap.add_argument("--report")
    a=ap.parse_args(); rows=read(ROOT/"provenance/experiment_catalog.json")["experiments"]
    if a.list:
        for r in rows: print(r["id"],r["manuscript_status"],r["question"],sep=" | ")
    if a.verify: verify()
    if a.report:
        row=next((r for r in rows if r["id"]==a.report),None)
        if row is None: ap.error("Unknown experiment; use --list")
        print(json.dumps(read(ROOT/row["summary"]),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
