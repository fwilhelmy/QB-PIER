# QB-PIER public release checklist

The `QB-PIER-public-release` tree is a **curated source-only export** from the uploaded snapshot.
It is not a security certification, a cleared publication, or a claim that the entire ML pipeline has been executed.

## Completed in this curated export

- [x] Excluded all original `experiments/` material, including downloaded MNIST files, trained weights, prototype notebooks, and output figures.
- [x] Excluded research-analysis notebooks/scripts and internal experiment queues.
- [x] Excluded site-specific `docs/HELPER.md` and the ÉTS-specific device profile.
- [x] Retained the training and FGSM evaluation entrypoints and required plotting helpers.
- [x] Removed the hardcoded Discord webhook URL; notifications default to disabled and read `DISCORD_WEBHOOK_URL` from the environment.
- [x] Expanded `.gitignore`, separated runtime/optional/dev dependencies, and supplied a static tree checker, CI checks, and lightweight hygiene tests.
- [x] Retained author, thesis, AI tooling, and library acknowledgments.

## Mandatory owner actions before going public

- [ ] **Revoke/rotate the previously embedded Discord webhook in Discord.** Deleting the hardcoded string does not revoke the credential.
- [ ] **Audit the original Git history and all remote refs** (including Git LFS, tags, branches, forks, CI logs). This ZIP had no `.git` metadata. Avoid force pushing contaminated history as the new public repository.
- [ ] Confirm written publication-scope and MIT licensing permission with research supervisors/coauthors and any relevant ÉTS policy requirements.
- [ ] Examine derivative/tutorial sources for license notices and required attribution. The repository has a pre-existing MIT license; that does not alone establish permission to license all components.
- [ ] Install the full environment on Python 3.10/3.11 and run an actual MNIST one-epoch training and FGSM evaluation before calling it validated.
- [ ] Inspect the final public Git diff and run a history-aware secret scanner such as Gitleaks against the actual clone.
- [ ] Confirm that no unpublished scientific contributions or claims remain in code/docstrings/configs beyond what coauthors approved.

## Suggested release workflow

Prefer creating a **new public repository from this curated directory**, with a new initial commit after approvals.
Do not copy the original `.git` directory. Maintain the original research repository privately.

```bash
python scripts/check_public_release.py .
python -m pytest tests/test_release_hygiene.py tests/test_configs.py tests/test_project_layout.py tests/test_qdevice_config.py tests/test_analytics_imports.py
python src/train.py model=lenet dataset=mnist training=debug
# Review docs/attacking.md and use real checkpoints generated above for FGSM.
```

`requirements.txt` specifies core dependencies; `requirements-optional.txt` and `requirements-dev.txt` separate auxiliary tooling. These are **not pinned to exact installed versions**. Preserve a tested environment lockfile or environment manifest when preparing a fully reproducible scientific release.
