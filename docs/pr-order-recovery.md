# Pull Request Order Recovery Playbook

Use this when PRs were opened in the wrong order and GitHub now shows unrelated commits in each PR.

## 1) Identify your base branch and PR chain

Most teams use `main` as the base. Confirm local state first:

```bash
git fetch origin --prune
git log --oneline --graph --decorate --all -n 40
```

Write your branch chain from oldest to newest (example):

- `feature/a` (should merge first)
- `feature/b` (depends on `feature/a`)
- `feature/c` (depends on `feature/b`)

## 2) Fast fix (retarget PR base branches)

If code order is correct but PR targets are wrong, only change PR base targets:

- PR for `feature/a` targets `main`
- PR for `feature/b` targets `feature/a`
- PR for `feature/c` targets `feature/b`

After `feature/a` merges, retarget `feature/b` back to `main`, then do the same for `feature/c`.

## 3) Full fix (rebuild branch stack)

If commits themselves are on wrong branches, rebuild from the true base (`main`) and cherry-pick commits in correct order.

### A. Recreate the first branch

```bash
git checkout main
git pull --ff-only origin main
git checkout -B feature/a-fixed
# cherry-pick commits for A
```

### B. Recreate the second branch on top of first

```bash
git checkout feature/a-fixed
git checkout -B feature/b-fixed
# cherry-pick commits for B
```

### C. Recreate the third branch on top of second

```bash
git checkout feature/b-fixed
git checkout -B feature/c-fixed
# cherry-pick commits for C
```

Use `git log --oneline --graph` after each step to verify commit order.

## 4) Validate each branch before force-pushing

```bash
git range-diff origin/main...feature/a origin/main...feature/a-fixed
git range-diff feature/a-fixed...feature/b feature/a-fixed...feature/b-fixed
git range-diff feature/b-fixed...feature/c feature/b-fixed...feature/c-fixed
```

Resolve differences until each fixed branch matches intended changes.

## 5) Update remote branches and PRs

```bash
git push -u origin feature/a-fixed
git push -u origin feature/b-fixed
git push -u origin feature/c-fixed
```

Open replacement PRs (recommended) or force-update existing branches:

```bash
git push --force-with-lease origin feature/a
git push --force-with-lease origin feature/b
git push --force-with-lease origin feature/c
```

Prefer replacement PRs if reviewers already left detailed comments on old PRs.

## 6) Prevent this in the future

- Create each new feature branch from the previous feature branch only when intentionally stacking.
- Name stacked branches with prefixes like `stack/01-a`, `stack/02-b`, `stack/03-c`.
- Run this check before opening a PR:

```bash
git log --oneline --decorate origin/main..HEAD
```

If commits from another feature appear, your branch base is wrong.
