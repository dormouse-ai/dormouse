"""The fused kernels, in one package so a call site says `fused.` rather than a path.

Nothing here is a default. `fused.residual_norm` with the flag off is the eager
expression it replaces, op for op, which is what makes adoption a one-line change at a
call site and what keeps the quality baseline unmoved while the kernel is measured.
"""
