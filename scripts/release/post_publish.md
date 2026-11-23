Post-publish checklist for {name} {version} (tag {tag}), repository {owner}/{name}.
Each item is the OWNER's action; the script never does these for you.

1. Private vulnerability reporting (SECURITY.md tells reporters to use it, so it must be on):
   https://github.com/{owner}/{name}/settings/security_analysis  ->  "Private vulnerability reporting"  ->  Enable.
2. Branch protection on main:
   https://github.com/{owner}/{name}/settings/branches  ->  Add rule for `main`: require a pull request, require the CI
   checks to pass, block force pushes and deletion.
3. Zenodo DOI (it archives a release only if the repository is switched on BEFORE the release is created; if this release
   already exists, switch the repository on, then edit the release and re-publish it, or cut the next release):
   https://zenodo.org/account/settings/github/  ->  flip the switch for {owner}/{name}.  Afterwards put the DOI in
   CITATION.cff and the README in a new commit (a pull request).  Creator: "Saltwater Research LLC" as an organisation
   (docs/RELEASING.md, Zenodo section).
4. Software Heritage archive:
   https://archive.softwareheritage.org/save/  ->  origin type "git", URL https://github.com/{owner}/{name}  ->  Submit.
5. Wayback Machine snapshots (repository, then the package pages):
   https://web.archive.org/save/https://github.com/{owner}/{name}
   https://web.archive.org/save/https://pypi.org/project/{name}/
   https://web.archive.org/save/https://crates.io/crates/{name}
6. Check what users will see:
   https://pypi.org/project/{name}/{version}/   https://crates.io/crates/{name}   https://github.com/{owner}/{name}/releases/tag/{tag}
   In a fresh virtual environment: python -m pip install "{name}=={version}"  then  python -c "import {name}; print({name}.version())"
7. Revoke the upload tokens you created for this release (PyPI, crates.io), or scope them to this project, and create new
   ones for the next release.
8. arXiv (cs.DC), when you are endorsed: https://arxiv.org/submit  ->  new submission.  Endorsement help:
   https://info.arxiv.org/help/endorsement.html .  arXiv shows the submitter's real name, whatever the paper's author line says.
   Cite the Zenodo DOI and the tag in the paper.
