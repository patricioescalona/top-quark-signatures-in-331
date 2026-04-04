from __future__ import annotations

import argparse
import cmath
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AsymmetricUncertainty:
    central: float
    minus: float
    plus: float


LAMBDA = AsymmetricUncertainty(0.22501, 0.00068, 0.00068)
A_PARAM = AsymmetricUncertainty(0.826, 0.015, 0.016)
RHO_BAR = AsymmetricUncertainty(0.1591, 0.0094, 0.0094)
ETA_BAR = AsymmetricUncertainty(0.3523, 0.0071, 0.0073)

REFERENCE_MODULI = (
    (
        0.97435,
        0.22501,
        0.003732,
    ),
    (
        0.22487,
        0.97349,
        0.04183,
    ),
    (
        0.00858,
        0.04111,
        0.99918,
    ),
)

RECOMMENDED_CONVENTION = (
    "Use the standard CKM parameterization with central values "
    "(lambda, A, rho_bar, eta_bar), converting rho_bar and eta_bar to rho and eta."
)


def to_rho_eta(lambda_value: float, rho_bar_value: float, eta_bar_value: float) -> tuple[float, float]:
    factor = 1.0 - lambda_value**2 / 2.0
    return rho_bar_value / factor, eta_bar_value / factor


def leading_order_ckm(lambda_value: float, a_value: float, rho_value: float, eta_value: float) -> tuple[tuple[complex, ...], ...]:
    return (
        (
            1.0 - lambda_value**2 / 2.0,
            lambda_value,
            a_value * lambda_value**3 * (rho_value - 1j * eta_value),
        ),
        (
            -lambda_value,
            1.0 - lambda_value**2 / 2.0,
            a_value * lambda_value**2,
        ),
        (
            a_value * lambda_value**3 * (1.0 - rho_value - 1j * eta_value),
            -a_value * lambda_value**2,
            1.0,
        ),
    )


def standard_ckm_from_wolfenstein(lambda_value: float, a_value: float, rho_value: float, eta_value: float) -> tuple[tuple[complex, ...], ...]:
    s12 = lambda_value
    s23 = a_value * lambda_value**2
    s13 = a_value * lambda_value**3 * math.sqrt(rho_value**2 + eta_value**2)
    delta = math.atan2(eta_value, rho_value)
    c12 = math.sqrt(1.0 - s12**2)
    c23 = math.sqrt(1.0 - s23**2)
    c13 = math.sqrt(1.0 - s13**2)
    phase = cmath.exp(-1j * delta)

    return (
        (
            c12 * c13,
            s12 * c13,
            s13 * phase,
        ),
        (
            -s12 * c23 - c12 * s23 * s13 / phase,
            c12 * c23 - s12 * s23 * s13 / phase,
            s23 * c13,
        ),
        (
            s12 * s23 - c12 * c23 * s13 / phase,
            -c12 * s23 - s12 * c23 * s13 / phase,
            c23 * c13,
        ),
    )


def modulus_matrix(matrix: tuple[tuple[complex, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(abs(entry) for entry in row) for row in matrix)


def sample_asymmetric(uncertainty: AsymmetricUncertainty, rng: random.Random) -> float:
    sigma = uncertainty.minus if rng.random() < 0.5 else uncertainty.plus
    return rng.gauss(uncertainty.central, sigma)


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentiles of an empty sample.")

    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def monte_carlo_bands(
    calculator,
    *,
    samples: int,
    seed: int,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    rng = random.Random(seed)
    collected = [[[] for _ in range(3)] for _ in range(3)]

    for _ in range(samples):
        lambda_value = sample_asymmetric(LAMBDA, rng)
        a_value = sample_asymmetric(A_PARAM, rng)
        rho_bar_value = sample_asymmetric(RHO_BAR, rng)
        eta_bar_value = sample_asymmetric(ETA_BAR, rng)
        rho_value, eta_value = to_rho_eta(lambda_value, rho_bar_value, eta_bar_value)
        moduli = modulus_matrix(calculator(lambda_value, a_value, rho_value, eta_value))
        for row_index in range(3):
            for col_index in range(3):
                collected[row_index][col_index].append(moduli[row_index][col_index])

    result = []
    for row in collected:
        formatted_row = []
        for values in row:
            values.sort()
            central = percentile(values, 0.5)
            minus = central - percentile(values, 0.158655)
            plus = percentile(values, 0.841345) - central
            formatted_row.append((central, minus, plus))
        result.append(tuple(formatted_row))
    return tuple(result)


def max_abs_difference(
    matrix_a: tuple[tuple[float, ...], ...],
    matrix_b: tuple[tuple[float, ...], ...],
) -> float:
    return max(
        abs(matrix_a[row_index][col_index] - matrix_b[row_index][col_index])
        for row_index in range(3)
        for col_index in range(3)
    )


def print_float_matrix(title: str, matrix: tuple[tuple[float, ...], ...]) -> None:
    print(title)
    for row in matrix:
        print("  " + "  ".join(f"{value:.6f}" for value in row))
    print()


def format_complex(value: complex) -> str:
    return f"{value.real:.8f}{value.imag:+.8f}j"


def print_complex_matrix(title: str, matrix: tuple[tuple[complex, ...], ...]) -> None:
    print(title)
    for row in matrix:
        print("  " + "  ".join(format_complex(value) for value in row))
    print()


def print_bands(title: str, bands: tuple[tuple[tuple[float, float, float], ...], ...]) -> None:
    print(title)
    for row in bands:
        pieces = []
        for central, minus, plus in row:
            pieces.append(f"{central:.6f} (-{minus:.6f}, +{plus:.6f})")
        print("  " + "  ".join(pieces))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CKM matrices from the image formulas and compare them with the quoted modulus values."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Monte Carlo samples used for the uncertainty propagation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=331,
        help="Random seed used for the Monte Carlo propagation.",
    )
    args = parser.parse_args()

    rho_value, eta_value = to_rho_eta(LAMBDA.central, RHO_BAR.central, ETA_BAR.central)
    recommended_matrix = standard_ckm_from_wolfenstein(
        LAMBDA.central, A_PARAM.central, rho_value, eta_value
    )

    leading_moduli = modulus_matrix(
        leading_order_ckm(LAMBDA.central, A_PARAM.central, rho_value, eta_value)
    )
    standard_moduli = modulus_matrix(recommended_matrix)

    print("Using the image inputs")
    print(f"  lambda  = {LAMBDA.central}")
    print(f"  A       = {A_PARAM.central}")
    print(f"  rho_bar = {RHO_BAR.central}")
    print(f"  eta_bar = {ETA_BAR.central}")
    print(f"  rho     = {rho_value:.6f}")
    print(f"  eta     = {eta_value:.6f}")
    print()

    print("Recommended choice")
    print(f"  {RECOMMENDED_CONVENTION}")
    print("  This is the one to use as your single numerical complex CKM matrix.")
    print()

    print_complex_matrix("Recommended complex CKM matrix:", recommended_matrix)

    print_float_matrix("Quoted |V_CKM| from the third image:", REFERENCE_MODULI)
    print_float_matrix(
        "Leading-order |V_CKM| from the first image formula:", leading_moduli
    )
    print_float_matrix(
        "Standard-parameterization |V_CKM| inferred from the same inputs:", standard_moduli
    )

    print(
        "Largest absolute difference with the quoted matrix:"
        f"\n  leading-order check : {max_abs_difference(leading_moduli, REFERENCE_MODULI):.6f}"
        f"\n  standard check      : {max_abs_difference(standard_moduli, REFERENCE_MODULI):.6f}"
    )
    print()

    print("Interpretation")
    print("  The first image is a truncated Wolfenstein expansion.")
    print("  The second image gives rho_bar and eta_bar, not rho and eta directly.")
    print("  The third-image moduli are matched much better by the recommended standard CKM matrix.")
    print()

    print_bands(
        "Monte Carlo 68% bands for the leading-order modulus matrix:",
        monte_carlo_bands(leading_order_ckm, samples=args.samples, seed=args.seed),
    )


if __name__ == "__main__":
    main()
