# Local-DDx Project Site

This folder contains the GitHub Pages site for Local-DDx, deployed automatically from the `docs/` folder on the `main` branch.

## Adding a Team Member

### Step 1: Add the CV

Place the PDF in `docs/cvs/` using lowercase-kebab-case:

```
docs/cvs/firstname-lastname.pdf
```

### Step 2: Add the HTML card

Open `docs/index.html` and find the `<!-- TEAM SECTION -->` comment. Inside the `<div class="team-grid">`, paste this block:

```html
<div class="team-card">
  <div class="team-avatar" aria-hidden="true">FL</div>
  <h3>First Last</h3>
  <p class="team-role">Their Role</p>
  <a href="cvs/firstname-lastname.pdf" class="team-cv-link" download>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
    Download CV
  </a>
</div>
```

Replace `FL` with initials, fill in name/role, update the `href`. Commit and push — the grid auto-accommodates any number of cards.

## File Structure

```
docs/
  index.html          The entire site (single page)
  css/style.css       All styles
  js/main.js          Nav scroll behavior, mobile menu
  cvs/                Team member CVs (PDF)
  assets/             Logo, OG image
  README.md           This file
```

## Editing

All content is in `index.html`. Styles are in one CSS file. JS is ~55 lines. No build tools, no frameworks, no dependencies.

Colors and spacing are controlled by CSS variables at the top of `style.css`.
