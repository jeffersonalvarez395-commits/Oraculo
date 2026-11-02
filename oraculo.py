import io, time, requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
import os
ENVIAR = True
TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT = os.environ.get("TELEGRAM_CHAT", "")
HORAS = 72
MAXP = 10
FILTRO = []
ETA, RHO = 0.02, 0.005
LIGAS = ["E0", "E1", "D1", "D2", "SP1", "SP2", "I1", "I2", "F1", "F2", "N1", "B1", "P1", "T1", "G1", "SC0"]
TEMPORADAS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
URLS = [(lg, f"https://www.football-data.co.uk/mmz4281/{t}/{lg}.csv") for t in TEMPORADAS for lg in LIGAS]
RESP = [(lg, requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)) for lg, u in URLS]
OK = [(lg, r) for lg, r in RESP if r.status_code == 200]
print(len(OK), "archivos de historial descargados")
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
mo = pd.concat([pd.read_csv(io.StringIO(r.content.decode("latin-1")), on_bad_lines="skip").reindex(columns=COLS).assign(league=lg) for lg, r in OK], ignore_index=True)
f = pd.to_datetime(mo.Date, format="%d/%m/%Y", errors="coerce")
mo["fecha"] = f.fillna(pd.to_datetime(mo.Date, format="%d/%m/%y", errors="coerce"))
mo[COLS[3:]] = mo[COLS[3:]].apply(pd.to_numeric, errors="coerce")
mo = mo.dropna(subset=["fecha", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).sort_values("fecha").reset_index(drop=True)
mo["gl"] = mo.FTHG
mo["ga"] = mo.FTAG
P = {"eta": ETA, "rho": RHO}
att, dfn, hadv, base = {}, {}, {}, {}
rows = list(zip(mo.league, mo.HomeTeam, mo.AwayTeam, mo.gl, mo.ga))
def paso(r): lg, h, a, gh, gv = r; e, rh = P["eta"], P["rho"]; ah, dh, aa_, da_ = att.get(h, 0.0), dfn.get(h, 0.0), att.get(a, 0.0), dfn.get(a, 0.0); hv, b = hadv.get(lg, 0.25), base.get(lg, 0.2); lh = np.exp(np.clip(b + hv + ah - da_, -3, 2)); lv = np.exp(np.clip(b + aa_ - dh, -3, 2)); eh, ev_ = gh - lh, gv - lv; att[h] = ah * (1 - rh) + e * eh; dfn[a] = da_ * (1 - rh) - e * eh; att[a] = aa_ * (1 - rh) + e * ev_; dfn[h] = dh * (1 - rh) - e * ev_; hadv[lg] = hv + e / 5 * eh; base[lg] = b + e / 5 * (eh + ev_) / 2; return (lh, lv)
_ = [paso(r) for r in rows]
print(len(mo), "partidos de historial | ratings listos para", len(att), "equipos | último partido:", mo.fecha.max().date())
ahora = pd.Timestamp.now(tz="America/Bogota")
rf = requests.get("https://www.football-data.co.uk/fixtures.csv", headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
fx = pd.read_csv(io.StringIO(rf.content.decode("latin-1")), on_bad_lines="skip")
fx.columns = [c.replace("ï»¿", "") for c in fx.columns]
fx = fx.dropna(subset=["Div", "Date", "Time", "HomeTeam", "AwayTeam"]).copy()
fx["dt"] = pd.to_datetime(fx["Date"] + " " + fx["Time"], format="%d/%m/%Y %H:%M", errors="coerce").dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("America/Bogota")
print("Fixtures leídos:", len(fx), "| fechas válidas:", fx["dt"].notna().sum(), "| rango:", fx["dt"].min(), "->", fx["dt"].max())
fx = fx[(fx["dt"] > ahora) & (fx["dt"] < ahora + pd.Timedelta(hours=HORAS))].copy()
fx = fx[fx.Div.isin(FILTRO)] if FILTRO else fx
fx = fx[fx.HomeTeam.isin(list(att)) & fx.AwayTeam.isin(list(att)) & fx.Div.isin(list(base))].sort_values("dt").copy()
print(len(fx), "partidos con datos en las próximas", HORAS, "horas")
fx["lh"] = [np.exp(np.clip(base[d] + hadv[d] + att[h] - dfn[a], -3, 2)) for d, h, a in zip(fx.Div, fx.HomeTeam, fx.AwayTeam)]
fx["lv"] = [np.exp(np.clip(base[d] + att[a] - dfn[h], -3, 2)) for d, h, a in zip(fx.Div, fx.HomeTeam, fx.AwayTeam)]
ks = np.arange(0, 10)
M3 = poisson.pmf(ks[None, :], fx.lh.values[:, None])[:, :, None] * poisson.pmf(ks[None, :], fx.lv.values[:, None])[:, None, :]
gi, gj = ks[:, None], ks[None, :]
t1, tx, t2 = (M3 * (gi > gj)).sum(axis=(1, 2)), (M3 * (gi == gj)).sum(axis=(1, 2)), (M3 * (gi < gj)).sum(axis=(1, 2))
fx["p1"], fx["px"], fx["p2"] = t1 / (t1 + tx + t2), tx / (t1 + tx + t2), t2 / (t1 + tx + t2)
mu = fx.lh + fx.lv
fx["p_over"] = 1 - np.exp(-mu) * (1 + mu + mu * mu / 2)
fx["p_btts"] = (1 - np.exp(-fx.lh)) * (1 - np.exp(-fx.lv))
def col(n): return pd.to_numeric(fx[n], errors="coerce") if n in fx.columns else pd.Series(np.nan, index=fx.index)
u1 = col("AvgH").notna() & col("AvgD").notna() & col("AvgA").notna()
fx["o1"], fx["ox"], fx["o2"] = np.where(u1, col("AvgH"), col("B365H")), np.where(u1, col("AvgD"), col("B365D")), np.where(u1, col("AvgA"), col("B365A"))
fx["fuente"] = np.where(u1, "promedio", "Bet365")
inv = 1 / fx.o1 + 1 / fx.ox + 1 / fx.o2
fx["m1"], fx["mxx"], fx["m2"] = (1 / fx.o1) / inv, (1 / fx.ox) / inv, (1 / fx.o2) / inv
u2 = col("Avg>2.5").notna() & col("Avg<2.5").notna()
fx["ov"] = np.where(u2, col("Avg>2.5"), col("B365>2.5"))
fx["un"] = np.where(u2, col("Avg<2.5"), col("B365<2.5"))
fx["pm_over"] = (1 / fx.ov) / (1 / fx.ov + 1 / fx.un)
fx["dif"] = np.abs(np.c_[fx.p1 - fx.m1, fx.px - fx.mxx, fx.p2 - fx.m2]).max(axis=1)
def pc(v): return "s/d" if pd.isna(v) else f"{v * 100:.0f}%"
def cu(v): return "s/d" if pd.isna(v) else f"{v:.2f}"
def lectura(d): return "sin cuotas para comparar" if pd.isna(d) else ("modelo y mercado coinciden" if d < 0.05 else ("discrepan algo: pesa más el mercado" if d < 0.12 else "discrepan mucho: lo más probable es que el modelo no vea algo que el mercado sí"))
def tarjeta(r): return (
f"⚽ {r.Div} · {r.HomeTeam} vs {r.AwayTeam}\n"
f"🕒 {r.dt.strftime('%d/%m %I:%M %p')} (hora Colombia)\n"
f"🎯 Goles esperados (modelo): {r.lh:.2f} - {r.lv:.2f}\n"
f"📊 1X2 modelo {pc(r.p1)}/{pc(r.px)}/{pc(r.p2)} | mercado {pc(r.m1)}/{pc(r.mxx)}/{pc(r.m2)}\n"
f"🏦 Cuotas 1X2: {cu(r.o1)} / {cu(r.ox)} / {cu(r.o2)} ({r.fuente})\n"
f"⬆️ Over 2.5: modelo {pc(r.p_over)} | mercado {pc(r.pm_over)} (cuota {cu(r.ov)})\n"
f"🤝 Ambos marcan (modelo): {pc(r.p_btts)}\n"
f"📌 Lectura: {lectura(r.dif)}"
)
cab = f"🔮 Oráculo · {len(fx)} partidos con datos en las próximas {HORAS} h (muestro {min(len(fx), MAXP)})\nReferencia informativa, no recomendaciones. En 17.903 partidos de prueba el mercado fue más preciso que este modelo (error 1X2: 0,9875 vs 1,0113), y lo mismo pasó en 16 ligas fuera de Europa."
textos = [cab] + [tarjeta(r) for r in fx.head(MAXP).itertuples()]
print("\n\n".join(textos))
ENV = [(requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", json={"chat_id": CHAT, "text": t}).json().get("ok"), time.sleep(1)) for t in textos] if (ENVIAR and TOKEN and len(fx) > 0) else []
