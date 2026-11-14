"""Rebuild data/adrs_raw.json from the three ADR repositories.
Clone them into ./sources first (see data/SOURCES.md), e.g.
  git clone --depth 1 https://github.com/ministryofjustice/cloud-platform sources/ministryofjustice_cloud-platform
"""
import re, json, glob, os, sys
SRC_DIR = os.environ.get("ADR_SRC", "sources")
SRC = {
 "cloud-platform": ("ministryofjustice_cloud-platform/architecture-decision-record", "https://github.com/ministryofjustice/cloud-platform/blob/main/architecture-decision-record/"),
 "modernisation-platform": ("ministryofjustice_modernisation-platform/architecture-decision-record", "https://github.com/ministryofjustice/modernisation-platform/blob/main/architecture-decision-record/"),
 "govuk-aws": ("alphagov_govuk-aws/docs/architecture/decisions", "https://github.com/alphagov/govuk-aws/blob/master/docs/architecture/decisions/"),
}
out=[]
for proj,(d,url) in SRC.items():
    for f in sorted(glob.glob(f"{SRC_DIR}/{d}/*.md")):
        name=os.path.basename(f)
        m=re.match(r"0*(\d+)[-_]", name)
        if not m: continue
        num=int(m.group(1))
        txt=open(f,encoding="utf-8").read()
        title=re.search(r"^#\s+(.+)$",txt,re.M)
        title=title.group(1).strip() if title else name
        title=re.sub(r"^\d+\.\s*","",title)
        date=re.search(r"Date:\s*([^\n]+)",txt)
        st=re.search(r"## Status\s*\n+(.+?)(?=\n## |\Z)",txt,re.S)
        status=st.group(1).strip() if st else ""
        links=re.findall(r"\]\(\.?/?(\d{3,4})[^)]*\.md\)",status)
        out.append(dict(id=f"{proj}:{num}",project=proj,num=num,title=title,date=date.group(1).strip() if date else "",status=status,status_links=[int(x) for x in links],url=url+name,text=txt))
json.dump(out,open(os.path.join(os.path.dirname(__file__), "..", "data", "adrs_raw.json"),"w"),indent=1)
print(len(out))
for a in out: print(a["id"],"|",a["title"],"|",a["status"].split("\n")[0][:70])
