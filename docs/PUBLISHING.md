# Publishing to your GitHub account

Extract the ZIP and open a terminal inside `apple-defect-attention`, the folder containing README.md. Create an empty repository in your GitHub account, then use its actual URL below:

```powershell
git init
git add .
git status
git commit -m "Add apple defect attention study"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/apple-defect-attention.git
git push -u origin main
```

Replace `YOUR_USERNAME`; do not put credentials or access tokens into the URL or repository. Review the staged file list before committing. This package has not been uploaded and contains no existing `.git` folder.

## Before making it public

- Start with a private repository while coursework is being assessed, and check course rules before publishing a complete solution.
- Confirm your coauthor agrees to publishing their name, report and slides.
- Inspect the final Word report and presentation. Author names and university affiliation are intentional; course handouts, private correspondence and obsolete drafts are excluded.
- Choose a software license only after checking ownership and compatibility with third-party dependencies. No MIT or other blanket license has been added automatically.
- Large checkpoints and datasets are excluded and ignored. Keep the complete private ZIP as your backup; do not commit it inside this repository.
- The final report still says no public repository is available. After publication, update that statement and add your real repository link if appropriate. Do not invent a URL in advance.

The Word report is the polished final version. The PowerPoint is the last prepared presentation; packaging does not silently edit either artifact. No final-report PDF or recorded video was available for this package.
