"""Generic entry point for residual-policy perceptual and ensemble audits.

The implementation remains in the original continuous-alpha audit module so
that existing commands stay reproducible while newer policies can share the
same validated evaluation code.
"""

from __future__ import annotations

from s6_audit_continuous_alpha_tail_refiner import main


if __name__ == "__main__":
    main()
