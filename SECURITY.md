# Security Policy

## Project status

**This project is experimental and unsupported.**

- We do **not** provide security guarantees.
- We do **not** commit to security patch SLAs or backports.
- We may have limited bandwidth to investigate or respond.

## Supported versions

Only the **latest commit on `main`** is considered for any security-related fixes (if any are made). Older commits, forks, and downstream deployments are **not supported**.

## Responsible use and liability

This project may be used for camera capture and license plate detection. You are responsible for using it lawfully and ethically (including respecting privacy and applicable local regulations).

**No liability**: the authors and contributors are not responsible for any misuse of this project or for any damages arising from its use. Use at your own risk. For the full terms, see `LICENSE`.

## Reporting a vulnerability

Please **do not** open public GitHub issues or pull requests for security vulnerabilities.

Use one of the following:

1. **GitHub “Report a vulnerability”** (preferred, if available in this repository’s *Security* tab).
2. If private reporting is not available, open a GitHub issue with the **minimum details needed** to contact you and coordinate privately (avoid posting exploit details, secrets, tokens, or sensitive logs).

When reporting, include:

- A clear description of the issue and potential impact
- Steps to reproduce (proof-of-concept if possible)
- Affected components (`camera/`, `server/`, scripts, etc.)
- Any mitigations/workarounds you’ve identified

## What to expect

- **Acknowledgement**: best-effort only.
- **Fixes**: may be provided as patches to `main`, or we may recommend mitigation/upgrade instead.
- **Disclosure**: please coordinate before public disclosure. We may be unable to meet specific timelines.

## Scope

In scope:

- Vulnerabilities in code in this repository.

Out of scope (typical examples):

- Vulnerabilities in third-party dependencies without a demonstrated impact here
- Misconfiguration or insecure deployment practices outside this repo
- DoS reports without clear, actionable reproduction and impact

