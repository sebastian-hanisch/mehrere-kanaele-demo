"""Abbildungen: alle Achsen fest (fixedrange), Ebene mit festem Bereich, Balken mit Zahl am Balken, Verlauf mit Schwelle."""

import numpy as np

import mkn_constants as C
import mkn_corr as R
import mkn_scenario as SC
import mkn_visualization as V


def _fest(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_verlauf_hat_feste_achsen_schwelle_und_alarmmarke():
    traj = SC.run_trajectory(5, 5, 3.0, 20, 35, C.KARTE_T2)
    fig = V.build_trajectory(traj, "T²")
    assert _fest(fig) and traj.alarm_t is not None
    namen = [t.name for t in fig.data]
    assert "erster Alarm" in namen and "Änderung" in namen and any((n or "").startswith("Schwelle") for n in namen)
    assert fig.layout.yaxis.range[1] > traj.threshold


def test_verlauf_ohne_aenderung_im_bild_hat_keine_aenderungslinie():
    traj = SC.run_trajectory(5, 1, 1.0, 0, 35, C.KARTE_SHEW)
    assert "Änderung" not in [t.name for t in V.build_trajectory(traj, "x").data]


def test_ebene_hat_feste_achsen_bereich_und_ellipse_im_bereich():
    fig = V.build_plane(0.95, 11.829, 3.2, 3.1, [1.0, -1.0])
    assert _fest(fig) and tuple(fig.layout.xaxis.range) == (-6, 6) and tuple(fig.layout.yaxis.range) == (-6, 6)
    ex, ey = R.ellipse_points(0.95, 11.829)
    assert max(np.abs(ex)) < 6 and max(np.abs(ey)) < 6


def test_balken_nennen_die_zahl_und_haben_lineare_achse():
    fig = V.build_bars([("A", 114.38, "#000"), ("B", 13.0, "#111")], "ARL₁")
    assert _fest(fig) and fig.layout.xaxis.type is None and list(fig.data[0].text) == ["114,38", "13,00"]
    assert fig.layout.yaxis.autorange == "reversed"


def test_strafe_und_verduennung_sind_logarithmisch_und_fest():
    fig = V.build_penalty((1, 2), {"T²": ("#000", [(1, 43.9), (2, 67.3)])}, 1.0)
    assert _fest(fig) and fig.layout.yaxis.type == "log"
    fig = V.build_dilution((1, 2), {"T²": ("#000", [(1, 17.9), (2, 17.9)])}, "ARL₁")
    assert _fest(fig) and fig.layout.yaxis.type == "log"
