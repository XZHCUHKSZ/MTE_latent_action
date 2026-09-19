"""Read-only release integrity and experiment catalogue inspector."""
import argparse, ast, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def verify():
    manifest=read(ROOT/"provenance/release_manifest.json")
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
    print(json.dumps({"status":"pass","files_checked":len(manifest["files"]),"scope":"Published integrity and syntax; no training or replay"}))
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
