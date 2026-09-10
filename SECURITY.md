# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in AutoProj, please report it
responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email: FeiChangShuai@users.noreply.github.com

Include the following in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 72 hours of receipt
- **Initial Assessment**: Within 7 days
- **Fix Release**: Depends on severity and complexity, typically within 30 days
- **Public Disclosure**: After fix is released, with credit to reporter (unless
  anonymity is requested)

## Supported Versions

Only the latest release line receives security updates.

| Version | Supported |
|---------|-----------|
| 2.1.x   | Yes       |
| < 2.1   | No        |

## Scope

Security vulnerabilities in the AutoProj Python package and its dependencies
are in scope. Issues in third-party libraries (NumPy, PyYAML, Numba) should be
reported to their respective maintainers.

## Out of Scope

- Theoretical vulnerabilities without proof of concept
- Social engineering attacks
- Denial of service through resource exhaustion (the library processes
  user-provided point cloud data by design)
