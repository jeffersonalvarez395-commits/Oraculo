# ORÁCULO — auditoría del motor v1

Incluye:
- `oraculo_engine.py`: funciones matemáticas puras, sin red ni Telegram.
- `tests/test_oraculo_engine.py`: pruebas unitarias.

Correcciones propuestas:
1. `rating_key(league, team)` separa equipos homónimos por liga.
2. `score_probabilities()` usa un soporte Poisson dinámico, no el corte fijo 0..9.
3. Over 2.5 y BTTS se calculan con expresiones Poisson exactas.
4. Cuotas 1X2 validadas y probabilidades normalizadas sin overround.
5. EV por unidad apostada: `p * cuota - 1`; no recomienda stake ni ejecuta apuestas.

Pruebas:
```bash
pip install numpy scipy
python -m unittest discover -s tests -v
```

IMPORTANTE: este módulo aún NO está integrado en `oraculo.py`. No se modificó
`main`, no se desplegó nada y no se envían mensajes a Telegram. La integración
debe cambiar todos los accesos a ratings para usar claves `(liga, equipo)`,
y debe hacerse después de revisar y ejecutar estas pruebas. Antes de usar
predicciones hay que realizar backtesting temporal y medir calibración,
Brier score, log loss y rendimiento neto.
