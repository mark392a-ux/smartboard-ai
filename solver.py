"""
solver.py — Parse recognized symbol sequence and solve with SymPy.
Produces step-by-step LaTeX output suitable for Streamlit's st.latex().
"""

from __future__ import annotations
import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from dataclasses import dataclass, field


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    raw_expression: str
    parsed_expression: str
    steps: list[dict]      # each: {"label": str, "latex": str, "explanation": str}
    result: str            # final answer as string
    result_latex: str      # LaTeX for the final answer
    is_equation: bool
    error: str | None = None
    is_numeric: bool = False


# ─── Symbol normalization ──────────────────────────────────────────────────────

SYMBOL_MAP = {
    '×': '*',
    '÷': '/',
    '−': '-',
    '–': '-',
    '—': '-',
    '^': '**',
    ',': '.',    # handle European decimal separator
}


def normalize_expression(raw: str) -> str:
    """Map handwriting-friendly symbols to Python/SymPy syntax."""
    expr = raw.strip()
    for src, dst in SYMBOL_MAP.items():
        expr = expr.replace(src, dst)
    # Remove spaces around operators for cleaner parsing
    expr = re.sub(r'\s+', '', expr)
    return expr


# ─── Main solver ──────────────────────────────────────────────────────────────

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def solve_expression(raw: str) -> SolverResult:
    """
    Parse `raw` (e.g. '3+4*2', '2x+6=0', '(3+4)*2') and return a SolverResult
    with step-by-step work in LaTeX.
    """
    normalized = normalize_expression(raw)
    steps: list[dict] = []

    # Check if it's an equation (contains '=')
    is_equation = '=' in normalized and normalized.count('=') == 1

    try:
        if is_equation:
            return _solve_equation(normalized, raw, steps)
        else:
            return _evaluate_expression(normalized, raw, steps)

    except Exception as e:
        return SolverResult(
            raw_expression=raw,
            parsed_expression=normalized,
            steps=steps,
            result="?",
            result_latex=r"\text{?}",
            is_equation=is_equation,
            error=f"Could not parse: {e}",
        )


# ─── Equation solver ──────────────────────────────────────────────────────────

def _solve_equation(normalized: str, raw: str, steps: list[dict]) -> SolverResult:
    lhs_str, rhs_str = normalized.split('=', 1)

    lhs = parse_expr(lhs_str, transformations=TRANSFORMATIONS)
    rhs = parse_expr(rhs_str, transformations=TRANSFORMATIONS)

    # Identify free symbols (variables)
    free = (lhs - rhs).free_symbols
    if not free:
        # No variable — evaluate both sides
        val_lhs = sp.simplify(lhs)
        val_rhs = sp.simplify(rhs)
        is_true = sp.simplify(val_lhs - val_rhs) == 0
        result_str = "True" if is_true else "False"
        steps.append({
            "label": "Check equality",
            "latex": f"{sp.latex(lhs)} = {sp.latex(rhs)} \\Rightarrow {result_str}",
            "explanation": "Both sides evaluated; checking if they are equal.",
        })
        return SolverResult(
            raw_expression=raw,
            parsed_expression=normalized,
            steps=steps,
            result=result_str,
            result_latex=f"\\text{{{result_str}}}",
            is_equation=True,
        )

    # Pick first variable (alphabetically)
    var = sorted(free, key=lambda s: str(s))[0]

    # Step 1: Show the equation
    steps.append({
        "label": "Original equation",
        "latex": f"{sp.latex(lhs)} = {sp.latex(rhs)}",
        "explanation": f"Solve for {var}.",
    })

    # Step 2: Rearrange → f(x) = 0
    equation = sp.Eq(lhs, rhs)
    expr = lhs - rhs
    simplified = sp.expand(expr)
    steps.append({
        "label": "Rearrange",
        "latex": f"{sp.latex(simplified)} = 0",
        "explanation": "Move all terms to the left-hand side.",
    })

    # Step 3: Factor if possible
    factored = sp.factor(simplified)
    if factored != simplified:
        steps.append({
            "label": "Factor",
            "latex": f"{sp.latex(factored)} = 0",
            "explanation": "Factor the expression.",
        })

    # Step 4: Solve
    solutions = sp.solve(equation, var)
    if not solutions:
        result_str = "No real solution"
        result_latex = r"\text{No real solution}"
    elif len(solutions) == 1:
        sol = solutions[0]
        result_str = str(sol)
        result_latex = f"{sp.latex(var)} = {sp.latex(sol)}"
        steps.append({
            "label": "Solution",
            "latex": result_latex,
            "explanation": f"The value of {var} that satisfies the equation.",
        })
    else:
        result_str = ", ".join(str(s) for s in solutions)
        result_latex = f"{sp.latex(var)} = " + ", ".join(sp.latex(s) for s in solutions)
        steps.append({
            "label": "Solutions",
            "latex": result_latex,
            "explanation": f"Multiple values of {var} satisfy the equation.",
        })

    # Step 5: Verification
    for sol in solutions[:2]:  # verify first two
        check_lhs = lhs.subs(var, sol)
        check_rhs = rhs.subs(var, sol)
        steps.append({
            "label": f"Verify: {var} = {sol}",
            "latex": (
                f"{sp.latex(lhs.subs(var, sol))} = {sp.latex(rhs.subs(var, sol))} "
                f"\\Rightarrow {sp.latex(sp.simplify(check_lhs))} = {sp.latex(sp.simplify(check_rhs))} "
                f"\\checkmark"
            ),
            "explanation": f"Substitute back to confirm the solution.",
        })

    return SolverResult(
        raw_expression=raw,
        parsed_expression=normalized,
        steps=steps,
        result=result_str,
        result_latex=result_latex,
        is_equation=True,
    )


# ─── Expression evaluator ─────────────────────────────────────────────────────

def _evaluate_expression(normalized: str, raw: str, steps: list[dict]) -> SolverResult:
    expr = parse_expr(normalized, transformations=TRANSFORMATIONS)

    # Step 1: Parse
    steps.append({
        "label": "Expression",
        "latex": sp.latex(expr),
        "explanation": "Parsed expression.",
    })

    # Step 2: Expand if beneficial
    expanded = sp.expand(expr)
    if expanded != expr:
        steps.append({
            "label": "Expand",
            "latex": sp.latex(expanded),
            "explanation": "Expand brackets.",
        })

    # Step 3: Simplify
    simplified = sp.simplify(expr)
    if simplified != expanded:
        steps.append({
            "label": "Simplify",
            "latex": sp.latex(simplified),
            "explanation": "Simplify the expression.",
        })

    # Step 4: Numeric evaluation
    try:
        numeric = float(simplified)
        is_int = numeric == int(numeric)
        result_str = str(int(numeric)) if is_int else f"{numeric:.6g}"
        result_latex = sp.latex(sp.nsimplify(simplified))
        steps.append({
            "label": "Result",
            "latex": f"= {result_latex}",
            "explanation": "Final numeric value.",
        })
        return SolverResult(
            raw_expression=raw,
            parsed_expression=normalized,
            steps=steps,
            result=result_str,
            result_latex=result_latex,
            is_equation=False,
            is_numeric=True,
        )
    except (TypeError, ValueError):
        # Symbolic result
        result_str = str(simplified)
        result_latex = sp.latex(simplified)
        steps.append({
            "label": "Simplified result",
            "latex": result_latex,
            "explanation": "Expression could not be evaluated numerically; leaving in symbolic form.",
        })
        return SolverResult(
            raw_expression=raw,
            parsed_expression=normalized,
            steps=steps,
            result=result_str,
            result_latex=result_latex,
            is_equation=False,
            is_numeric=False,
        )


# ─── Sequence builder ─────────────────────────────────────────────────────────

def labels_to_expression(labels: list[str]) -> str:
    """
    Convert a left-to-right list of predicted labels into an expression string.
    E.g. ['3', '+', '4', '×', '2'] → '3+4×2'
    """
    return "".join(labels)


def is_valid_expression(expr: str) -> bool:
    """Quick syntax check before heavy SymPy parsing."""
    if not expr.strip():
        return False
    normalized = normalize_expression(expr)
    try:
        if '=' in normalized:
            lhs, rhs = normalized.split('=', 1)
            parse_expr(lhs, transformations=TRANSFORMATIONS)
            parse_expr(rhs, transformations=TRANSFORMATIONS)
        else:
            parse_expr(normalized, transformations=TRANSFORMATIONS)
        return True
    except Exception:
        return False
