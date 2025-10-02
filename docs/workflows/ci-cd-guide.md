# CI/CD Guide

Complete guide to the Continuous Integration and Continuous Deployment setup for LLM-Local-Lab.

**Last Updated**: 2025-10-02

---

## Overview

This project uses **GitHub Actions** for automated testing, validation, benchmarking, and releases. All workflows are defined in `.github/workflows/`.

---

## Available Workflows

### 1. CI - Code Validation

**File**: `.github/workflows/ci-validation.yml`

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**Jobs**:

#### Lint and Format Check
- Runs `black` to check code formatting
- Runs `flake8` for linting
- Runs `mypy` for type checking

#### Test Scripts
- Validates Python script imports
- Checks for syntax errors
- Runs pytest (if tests exist)

#### Validate Configs
- Validates all YAML configuration files
- Checks `pyproject.toml` syntax

#### Check Documentation
- Verifies documentation structure
- Checks for broken markdown links
- Ensures README and CLAUDE.md exist

#### Security Check
- Scans dependencies for known vulnerabilities
- Checks for accidental secrets in code

**Expected Result**: ✅ All checks pass before merging PRs

---

### 2. Benchmark Report Generator

**File**: `.github/workflows/benchmark-report.yml`

**Triggers**:
- Manual dispatch (workflow_dispatch)
- Push of new benchmark results (`benchmarks/results/raw/*.json`)

**Jobs**:

#### Generate Report
- Finds latest benchmark result JSON
- Generates formatted markdown report
- Commits report to `benchmarks/results/reports/`
- Posts results as GitHub issue comment (optional)

**Usage**:
```bash
# Locally run a benchmark
python benchmarks/scripts/llm_inference_benchmark.py --model MODEL_NAME

# Commit the result
git add benchmarks/results/raw/*.json
git commit -m "[BENCHMARK]: Model Name results"
git push

# Workflow automatically generates report
```

**Manual Trigger**:
- Go to Actions tab in GitHub
- Select "Benchmark Report Generator"
- Click "Run workflow"
- Choose benchmark type

---

### 3. Release Automation

**File**: `.github/workflows/release.yml`

**Triggers**:
- Push of version tag (e.g., `v0.1.0`)
- Manual dispatch with version input

**Jobs**:

#### Create Release
- Extracts version from tag
- Generates changelog from commits
- Counts benchmark statistics
- Creates GitHub Release with notes

#### Build Artifacts
- Builds Python package
- Validates package integrity
- Uploads distribution files

#### Publish to PyPI (disabled by default)
- Can be enabled when ready to publish
- Requires PyPI credentials in GitHub secrets

**Usage**:
```bash
# Create and push a tag
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# Workflow automatically creates release
```

**Changelog Grouping**:
- ✨ Features: Commits starting with `feat` or `add`
- 🐛 Bug Fixes: Commits starting with `fix` or `bug`
- 📚 Documentation: Commits starting with `docs`
- 📊 Benchmarks: Commits with `[BENCHMARK]`
- 🔧 Other Changes: All other commits

---

## Badge Status

Add these badges to your README for live status:

```markdown
![CI Status](https://github.com/YOURUSERNAME/LLM-Local-Lab/workflows/CI%20-%20Code%20Validation/badge.svg)
![License](https://img.shields.io/github/license/YOURUSERNAME/LLM-Local-Lab)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.9+-red.svg)
![CUDA](https://img.shields.io/badge/cuda-13.0-green.svg)
```

---

## Local Development

### Running CI Checks Locally

Before pushing, run these checks locally:

```bash
# Format check
black --check .

# Auto-format
black .

# Lint
flake8 . --max-line-length=100

# Type check
mypy benchmarks/scripts/ utils/

# Syntax check
python -m py_compile benchmarks/scripts/*.py
```

### Pre-commit Hook (Recommended)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running pre-commit checks..."

# Format check
black --check . || {
    echo "Code formatting failed. Run 'black .' to fix."
    exit 1
}

# Lint critical errors
flake8 . --select=E9,F63,F7,F82 || {
    echo "Critical lint errors found."
    exit 1
}

echo "Pre-commit checks passed!"
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Workflow Customization

### Modify Trigger Conditions

Edit workflow files to change when they run:

```yaml
# Run on different branches
on:
  push:
    branches: [ main, develop, feature/* ]

# Run on specific paths
on:
  push:
    paths:
      - 'benchmarks/**'
      - 'models/**'

# Run on schedule (weekly)
on:
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday at midnight
```

### Add Custom Jobs

Example: Add GPU availability check

```yaml
gpu-check:
  name: Check GPU Availability
  runs-on: self-hosted  # Requires self-hosted runner with GPUs

  steps:
  - uses: actions/checkout@v4

  - name: Check CUDA
    run: |
      nvidia-smi
      python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Note**: GitHub-hosted runners don't have GPUs. You need self-hosted runners for GPU tasks.

---

## Self-Hosted Runners (Advanced)

To run benchmarks in CI/CD, you need a self-hosted runner with GPUs.

### Setup Self-Hosted Runner

1. **In GitHub**:
   - Settings → Actions → Runners → New self-hosted runner
   - Follow platform-specific instructions

2. **On Your Machine** (AMD 9950X + dual RTX 5090):
   ```bash
   # Download and configure runner
   mkdir actions-runner && cd actions-runner
   # Follow GitHub's download instructions

   # Install as service
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

3. **Add GPU Tag**:
   - Label runner as `gpu`, `rtx-5090`, or similar
   - Use in workflows: `runs-on: [self-hosted, gpu]`

### Security Considerations

⚠️ **Important**:
- Self-hosted runners can execute arbitrary code from PRs
- Only use for private repos or trusted contributors
- Isolate runner environment (Docker, VM)
- Never expose API keys or credentials

---

## Secrets Management

Store sensitive data in GitHub Secrets:

**Settings → Secrets and variables → Actions → New repository secret**

### Recommended Secrets

- `HUGGINGFACE_TOKEN`: For downloading gated models
- `OPENAI_API_KEY`: If using OpenAI for analysis
- `PYPI_API_TOKEN`: For publishing packages (if enabled)

**Access in workflows**:
```yaml
- name: Use secret
  env:
    HF_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
  run: |
    echo "Token is available as environment variable"
```

---

## Debugging Workflows

### View Logs

1. Go to **Actions** tab in GitHub
2. Click on specific workflow run
3. Click on job name
4. Expand step to see logs

### Enable Debug Logging

Add to repository secrets:
- `ACTIONS_RUNNER_DEBUG`: `true`
- `ACTIONS_STEP_DEBUG`: `true`

Re-run workflow to see detailed logs.

### Common Issues

**Issue**: Workflow not triggering
- **Check**: Branch names, file paths in trigger conditions
- **Check**: Workflow file syntax (YAML indentation)

**Issue**: Job failing on specific step
- **Check**: Logs for error messages
- **Check**: Dependencies are installed correctly
- **Try**: Run same command locally

**Issue**: Timeout
- **Fix**: Increase timeout in workflow:
  ```yaml
  jobs:
    my-job:
      timeout-minutes: 30  # Default is 360
  ```

---

## Best Practices

### 1. Fast Feedback

- Keep CI runs under 5 minutes when possible
- Use caching for dependencies
- Run expensive tests only on main branch

### 2. Fail Fast

- Run quick checks first (linting)
- Run slow checks last (full benchmarks)
- Use `continue-on-error: false` for critical checks

### 3. Clear Commit Messages

Follow conventional commits for better changelogs:

```
feat: add support for Llama 3.2 models
fix: resolve OOM error with 70B models
docs: update CI/CD guide
[BENCHMARK]: Mixtral 8x7B Q5 results
```

### 4. Semantic Versioning

Use semantic versioning for releases:
- `v1.0.0`: Major release (breaking changes)
- `v0.1.0`: Minor release (new features)
- `v0.0.1`: Patch release (bug fixes)

### 5. Test Locally First

Always run checks locally before pushing:
```bash
black . && flake8 . && python -m pytest
```

---

## Workflow Examples

### Example 1: Adding a New Benchmark

```bash
# 1. Run benchmark locally
python benchmarks/scripts/llm_inference_benchmark.py --model meta-llama/Llama-3.1-8B

# 2. Commit results
git add benchmarks/results/raw/*.json
git commit -m "[BENCHMARK]: Llama 3.1 8B FP16 results"

# 3. Push to GitHub
git push origin main

# 4. CI automatically:
#    - Validates code
#    - Generates report
#    - Commits report back
```

### Example 2: Creating a Release

```bash
# 1. Update version in pyproject.toml
# version = "0.2.0"

# 2. Commit changes
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0"

# 3. Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0: Llama 3.1 support"
git push origin v0.2.0

# 4. CI automatically:
#    - Generates changelog
#    - Creates GitHub release
#    - Builds package
```

### Example 3: Contributing via PR

```bash
# 1. Create feature branch
git checkout -b feature/add-qwen-support

# 2. Make changes
# ... edit files ...

# 3. Run local checks
black .
flake8 .

# 4. Commit and push
git add .
git commit -m "feat: add support for Qwen 2.5 models"
git push origin feature/add-qwen-support

# 5. Create PR on GitHub
# 6. CI runs automatically on PR
# 7. Merge after CI passes
```

---

## Future Enhancements

Potential CI/CD improvements:

- [ ] Automated weekly benchmark runs (scheduled)
- [ ] Performance regression detection
- [ ] Automated model downloads from HuggingFace
- [ ] Slack/Discord notifications for benchmark completion
- [ ] Automated comparison reports (new vs old benchmarks)
- [ ] GPU utilization metrics in CI
- [ ] Docker containerized benchmarks

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Self-Hosted Runners Guide](https://docs.github.com/en/actions/hosting-your-own-runners)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Action Marketplace](https://github.com/marketplace?type=actions)

---

**Questions or Issues?**

Open an issue using the "Feature Request" template to suggest CI/CD improvements.
