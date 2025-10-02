# GitHub Setup Instructions

Complete guide to creating the GitHub repository and pushing LLM-Local-Lab with full CI/CD.

**Date**: 2025-10-02

---

## Step 1: Create GitHub Repository

### Option A: Using GitHub CLI (Recommended)

```bash
# Install GitHub CLI if not already installed
# Windows: winget install GitHub.cli
# Or download from: https://cli.github.com/

# Login to GitHub
gh auth login

# Create repository (choose public or private)
cd C:\Users\Gamer\Dev\LLM-Local-Lab
gh repo create LLM-Local-Lab --public --source=. --remote=origin --push

# Repository is now created and code is pushed!
```

### Option B: Using GitHub Web Interface

1. **Go to GitHub**: https://github.com/new

2. **Repository Settings**:
   - **Name**: `LLM-Local-Lab`
   - **Description**: Professional LLM experimentation with dual RTX 5090 GPUs
   - **Visibility**: Public (or Private if preferred)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)

3. **Click**: "Create repository"

4. **Copy the repository URL** shown (e.g., `https://github.com/YOURUSERNAME/LLM-Local-Lab.git`)

---

## Step 2: Link Local Repository to GitHub

```bash
# Navigate to project
cd C:\Users\Gamer\Dev\LLM-Local-Lab

# Add GitHub remote (replace YOURUSERNAME with your GitHub username)
git remote add origin https://github.com/YOURUSERNAME/LLM-Local-Lab.git

# Verify remote
git remote -v
```

**Expected output**:
```
origin  https://github.com/YOURUSERNAME/LLM-Local-Lab.git (fetch)
origin  https://github.com/YOURUSERNAME/LLM-Local-Lab.git (push)
```

---

## Step 3: Update README Badges

Before pushing, update the badges in README.md with your actual username:

```bash
# Open README.md and replace YOURUSERNAME with your GitHub username
# Line 3: ![CI Status](https://github.com/YOURUSERNAME/LLM-Local-Lab/workflows/CI%20-%20Code%20Validation/badge.svg)
```

**Example**:
```markdown
![CI Status](https://github.com/johndoe/LLM-Local-Lab/workflows/CI%20-%20Code%20Validation/badge.svg)
```

After editing:
```bash
git add README.md
git commit -m "docs: update badges with GitHub username"
```

---

## Step 4: Commit CI/CD Files

```bash
# Add all CI/CD configuration files
git add .github/
git add LICENSE
git add docs/workflows/ci-cd-guide.md

# Commit
git commit -m "ci: add GitHub Actions workflows and templates

- CI validation workflow (linting, formatting, tests)
- Benchmark report generator
- Release automation
- Issue and PR templates
- CI/CD documentation

Workflows:
- ci-validation.yml: Code quality checks
- benchmark-report.yml: Auto-generate benchmark reports
- release.yml: Automated releases with changelog

Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Step 5: Push to GitHub

```bash
# Push to main branch
git push -u origin main
```

**Expected output**:
```
Enumerating objects: XX, done.
Counting objects: 100% (XX/XX), done.
...
To https://github.com/YOURUSERNAME/LLM-Local-Lab.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

## Step 6: Verify GitHub Setup

### Check Repository

1. Go to: `https://github.com/YOURUSERNAME/LLM-Local-Lab`
2. Verify:
   - ✅ README displays with badges
   - ✅ Directory structure is visible
   - ✅ License file is present

### Check Actions

1. Go to: **Actions** tab
2. Verify:
   - ✅ Workflows are listed:
     - CI - Code Validation
     - Benchmark Report Generator
     - Release
   - ✅ Initial push triggered CI workflow
   - ✅ CI workflow passed (or check errors)

### Check Issue Templates

1. Go to: **Issues** tab → **New issue**
2. Verify templates appear:
   - ✅ Benchmark Request
   - ✅ Bug Report
   - ✅ Feature Request

---

## Step 7: Configure Repository Settings (Optional)

### Enable GitHub Pages (for documentation)

1. **Settings** → **Pages**
2. **Source**: Deploy from a branch
3. **Branch**: `main`, folder: `/docs`
4. **Save**

### Branch Protection (Recommended)

1. **Settings** → **Branches** → **Add branch protection rule**
2. **Branch name pattern**: `main`
3. **Enable**:
   - ✅ Require status checks to pass (select: CI - Code Validation)
   - ✅ Require branches to be up to date
   - ✅ Require pull request reviews (if working with others)

### Secrets (if needed)

1. **Settings** → **Secrets and variables** → **Actions**
2. **New repository secret**
3. Add:
   - `HUGGINGFACE_TOKEN`: For downloading gated models (optional)
   - `OPENAI_API_KEY`: If using OpenAI for analysis (optional)

---

## Step 8: First Actions Test

### Trigger CI Workflow

```bash
# Make a small change to test CI
echo "# Test" >> README.md

git add README.md
git commit -m "test: trigger CI workflow"
git push

# Go to Actions tab and watch workflow run
```

### Expected CI Results

All jobs should pass:
- ✅ Lint and Format Check
- ✅ Test Scripts
- ✅ Validate Configs
- ✅ Check Documentation
- ✅ Security Check

If any fail, check logs and fix issues.

---

## Step 9: Create First Release (Optional)

```bash
# Ensure everything is committed
git status  # Should show "nothing to commit, working tree clean"

# Create and push tag
git tag -a v0.1.0 -m "Release v0.1.0: Initial LLM-Local-Lab setup"
git push origin v0.1.0

# Check Releases page on GitHub
# Automated release should be created with changelog
```

---

## Troubleshooting

### Issue: Push Rejected (Authentication)

**Error**: `remote: Permission denied`

**Fix**:
```bash
# Configure Git credentials
git config --global credential.helper store

# Or use SSH instead of HTTPS
git remote set-url origin git@github.com:YOURUSERNAME/LLM-Local-Lab.git

# Generate SSH key if needed
ssh-keygen -t ed25519 -C "your_email@example.com"
# Add public key to GitHub: Settings → SSH and GPG keys
```

### Issue: CI Workflow Not Running

**Possible causes**:
- Workflow file syntax error (check YAML indentation)
- Actions disabled in repository settings
- Branch name mismatch (check if branch is `main` not `master`)

**Fix**:
1. Go to **Settings** → **Actions** → **General**
2. Ensure "Allow all actions and reusable workflows" is selected
3. Check workflow file syntax: `yamllint .github/workflows/*.yml`

### Issue: Badge Not Showing

**Cause**: Repository is private or workflow hasn't run yet

**Fix**:
1. Make repository public, or
2. Generate workflow status badge from Actions tab
3. Ensure workflow has run at least once

---

## Next Steps After Setup

1. **Run First Benchmark**:
   ```bash
   python benchmarks/scripts/embedding_benchmark.py
   git add benchmarks/results/raw/*.json
   git commit -m "[BENCHMARK]: Initial embedding baseline"
   git push
   # Watch automatic report generation in Actions
   ```

2. **Create Development Branch**:
   ```bash
   git checkout -b develop
   git push -u origin develop
   # Work on develop, PR to main
   ```

3. **Invite Collaborators** (if applicable):
   - **Settings** → **Collaborators** → **Add people**

4. **Setup Self-Hosted Runner** (for GPU benchmarks):
   - See `docs/workflows/ci-cd-guide.md` → "Self-Hosted Runners" section

---

## Maintenance

### Keep CI/CD Updated

```bash
# Periodically update GitHub Actions
# Check for newer versions in:
# - actions/checkout
# - actions/setup-python
# - softprops/action-gh-release

# Update in .github/workflows/*.yml
```

### Monitor Actions Usage

- **Settings** → **Billing and plans** → **Actions**
- Free tier: 2,000 minutes/month for public repos
- Self-hosted runners: No minute limits

---

## Summary Checklist

After completing setup, verify:

- [ ] Repository created on GitHub
- [ ] Local repository linked to GitHub remote
- [ ] README badges updated with username
- [ ] CI/CD workflows committed and pushed
- [ ] LICENSE file present
- [ ] Main branch pushed successfully
- [ ] Actions tab shows workflows
- [ ] CI workflow passed on first push
- [ ] Issue templates visible
- [ ] PR template configured
- [ ] (Optional) Branch protection enabled
- [ ] (Optional) First release created

---

**Congratulations!** Your LLM-Local-Lab is now on GitHub with full CI/CD automation.

**Repository URL**: `https://github.com/YOURUSERNAME/LLM-Local-Lab`

**Next**: Start running benchmarks and let automation handle the rest!
