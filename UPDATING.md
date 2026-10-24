# Weekly website update

Instructions for the automated weekly update of https://jonmay.github.io/webpage/
(the source is `index.html` and `jonmay.bib` in this repo, published by GitHub Pages
from `master`). The job makes changes on a branch and opens a pull request; nothing
goes live until Jon merges it. Edit this file to change what the job does.

## Ground rules

- Never push to `master`. Work on a branch named `update/YYYY-MM-DD` and open a PR.
- If nothing needs changing, don't open a PR.
- Keep the hand-written house style: copy the formatting of neighboring entries exactly.
  Don't reformat, re-indent, or "modernize" anything you weren't asked to touch.
- Only add things verified from a primary source (the venue's site, ACL Anthology,
  OpenReview, PMLR, the outlet's own article). Never invent or guess a URL; fetch it
  and confirm it is the right item before adding it.
- Make sure it is this Jonathan May (USC Computer Science / USC ISI, NLP,
  director of CUTELABNAME), not someone else with the same name.
- One commit per kind of change (publications, arXiv upgrades, press, photos, links).
- Anything uncertain goes in the PR description under "Needs your check", not in the page.

## What to check each week

### 1. New publications
Look for papers co-authored by Jonathan May that are not yet in `index.html`:
DBLP (`https://dblp.org/search/publ/api?q=author:Jonathan_May:&format=json` and his
DBLP person page), the Semantic Scholar author API, ACL Anthology, OpenReview, arXiv.
- Peer-reviewed papers go under `<h2> Publications` in the right `<h3>YEAR</h3>`,
  newest venue first, in the format
  `"Title", A. Author, J. May. Proc. VENUE, YEAR. <a href="URL">Paper</a>.<br>`.
  Create a new `<h3>YEAR</h3>` heading when a new year starts.
- arXiv-only preprints go under `<h2> Non-Publications`, same style.
- Add a matching BibTeX entry to `jonmay.bib`, following its key convention.

### 2. arXiv upgrades
For every entry that links to arxiv.org or sits in Non-Publications, check whether it
has since been published at a conference or journal. If so, move it to Publications
under the right year, change the venue text, and point the link at the official
version (keep the arXiv link only if the entry already lists multiple links).
Update the BibTeX entry too.

### 3. Press
Find new press (news articles, interviews, podcasts, radio, TV, YouTube, USC/ISI news)
featuring or quoting Jon that is not in the `<h2> Press </h2>` list. Add newest first:
`<li> <a href="URL"> Outlet -- "Headline" (M/D/YY)</a> </li>`.

### 4. Lab photos
Add new lab outing / group photos to `<h2>Group Pics</h2>`, newest year on top, in
the same table-cell format as the existing rows. Sources:
- `imgs/incoming/` in this repo: Jon drops photos here (e.g. uploaded from his phone
  via GitHub's "Add file → Upload files"). Resize to at most 1800px on the long side,
  move each into `imgs/` as `YYYY_short_description.jpg`, and add it to the page.
  Use EXIF date if present, else the file's commit date. If a caption is not obvious,
  use a generic one ("Lab Outing") and ask in the PR.
- The lab Flickr account https://www.flickr.com/photos/183261364@N02/ (albums,
  including "CUTELABNAME photos"). Add photos of lab outings/events that aren't on
  the page yet; hotlink `live.staticflickr.com` images and link to the photo page,
  as the existing Flickr entries do.
Only add photos of lab events (outings, group pics, lunches, graduations).

### 5. Link health
Check every link in `index.html` (skip commented-out HTML). Many sites (ACM DL,
inc.com, isi.edu, quora) return 403 to bots; treat those as fine. For real breakage
(404, dead domain, missing local file): find the moved/official URL, or fall back to
a Wayback Machine snapshot (`https://archive.org/wayback/available?url=...`). List
anything you couldn't fix in the PR description.

### 6. CV and bios (report only)
`cv.pdf` and `bios.txt` are maintained by hand. Don't edit them. In the PR
description, list papers from the current and previous year that appear on the page
but not in `cv.pdf`, and say whether the newest bio in `bios.txt` looks stale
(title, roles, awards, number of PhD graduates).

## PR description

Summarize changes by section, with links, then "Needs your check" for anything
uncertain, then the CV/bios report.
