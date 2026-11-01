"""ORÁCULO: núcleo matemático puro v1. Sin red, secretos ni Telegram."""
import math
import unicodedata
from typing import Mapping
import numpy as np
from scipy.stats import poisson

DEFAULT_TAIL_TOLERANCE = 1e-12
MAX_LAMBDA = 100.0

def _label(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} debe ser texto no vacío")
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.casefold().split())

def rating_key(league: str, team: str) -> tuple[str, str]:
    """Evita compartir ratings de equipos homónimos entre ligas."""
    return _label(league, "league"), _label(team, "team")

def _valid_lambda(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser numérico") from None
    if not math.isfinite(value) or value < 0 or value > MAX_LAMBDA:
        raise ValueError(f"{name} debe estar entre 0 y {MAX_LAMBDA}")
    return value

def score_probabilities(lambda_home, lambda_away,
                        tail_tolerance=DEFAULT_TAIL_TOLERANCE):
    """Matriz de marcadores Poisson con soporte dinámico y cola controlada."""
    lh = _valid_lambda(lambda_home, "lambda_home")
    la = _valid_lambda(lambda_away, "lambda_away")
    try:
        tol = float(tail_tolerance)
    except (TypeError, ValueError):
        raise ValueError("tail_tolerance debe ser numérico") from None
    if not math.isfinite(tol) or not 0 < tol < 0.1:
        raise ValueError("tail_tolerance debe estar entre 0 y 0.1")

    q = 1.0 - tol / 2.0
    mh = int(poisson.ppf(q, lh)) if lh else 0
    ma = int(poisson.ppf(q, la)) if la else 0
    ph = poisson.pmf(np.arange(mh + 1), lh)
    pa = poisson.pmf(np.arange(ma + 1), la)
    matrix = np.outer(ph, pa)
    mass = float(matrix.sum())
    if not math.isfinite(mass) or mass <= 0:
        raise ArithmeticError("No se pudo normalizar la matriz")
    matrix /= mass

    p_home = float(np.tril(matrix, k=-1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, k=1).sum())
    norm = p_home + p_draw + p_away
    mu = lh + la

    return {
        "p_home": p_home / norm,
        "p_draw": p_draw / norm,
        "p_away": p_away / norm,
        "p_over_2_5": float(poisson.sf(2, mu)),
        "p_btts": float((1 - math.exp(-lh)) * (1 - math.exp(-la))),
        "score_matrix": matrix,
        "max_home_goals": mh,
        "max_away_goals": ma,
        "captured_mass": mass,
    }

def market_probabilities_1x2(odds: Mapping[str, float]):
    """Probabilidades implícitas 1X2 normalizadas para quitar overround."""
    implied = {}
    for key in ("home", "draw", "away"):
        if key not in odds:
            raise ValueError(f"Falta la cuota {key}")
        try:
            odd = float(odds[key])
        except (TypeError, ValueError):
            raise ValueError(f"Cuota {key} inválida") from None
        if not math.isfinite(odd) or odd <= 1:
            raise ValueError(f"Cuota {key} debe ser finita y mayor que 1")
        implied[key] = 1.0 / odd
    total = sum(implied.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Margen de mercado inválido")
    return {k: v / total for k, v in implied.items()}

def expected_value(probability, decimal_odds):
    """EV por unidad: p * cuota - 1. No calcula stake ni apuesta."""
    try:
        p, odd = float(probability), float(decimal_odds)
    except (TypeError, ValueError):
        raise ValueError("Probabilidad y cuota deben ser numéricas") from None
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Probabilidad debe estar entre 0 y 1")
    if not math.isfinite(odd) or odd <= 1:
        raise ValueError("Cuota debe ser finita y mayor que 1")
    return p * odd - 1.0
