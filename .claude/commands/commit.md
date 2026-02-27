Review the changes I've made, write a good commit message, and commit them.

## Steps

1. Run `git status` and `git diff` (staged + unstaged) to understand all changes.
2. Run `git log --oneline -5` to match the repo's commit message style.
3. Draft a concise commit message summarizing the "why" (not the "what").
4. Stage relevant files (prefer specific filenames over `git add -A`).
5. Commit with the message. End with: `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
6. Run `git status` to verify success.

## Rules

- Do NOT push unless explicitly asked.
- Do NOT commit files that likely contain secrets (.env, credentials.json, etc.).
- Do NOT amend existing commits — always create a new commit.
- If a pre-commit hook fails, fix the issue and create a NEW commit (do not --amend).
- Use a HEREDOC for the commit message to preserve formatting.
