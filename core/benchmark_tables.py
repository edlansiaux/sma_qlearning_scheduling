"""
core/benchmark_tables.py - Génération des Tableaux 1 et 2 conformes au diaporama.

Tableau 1 (sans collaboration) :
  Colonnes : AG | Tabou | RS | Agent_AG | Agent_Tabou | Agent_RS
  Lignes   : Jour 1..4 avec nombre de patients variable

Tableau 2 (avec collaboration) :
  Sous-tableaux : Sans Apprentissage / Avec Apprentissage
  Pour chaque : Amis (AG_Tabou, AG_RS, Tabou_RS, AG_Tabou_RS)
              | Ennemis (AG_Tabou, AG_RS, Tabou_RS, AG_Tabou_RS)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import SchedulingEnvironment, generate_random_data, DEFAULT_SKILLS, DEFAULT_MAX_OPS
from core.agents import MultiAgentSystem, CollaborationMode
from visualization import plot_gantt, plot_convergence

OUTPUT_DIR = "benchmark_results"
ITERATIONS = 50

# Jours de benchmark (reproduire les lignes du diapo)
DAYS = [
    {"day": 1, "num_patients": 10,  "seed": 100},
    {"day": 2, "num_patients": 68,  "seed": 200},
    {"day": 3, "num_patients": 78,  "seed": 300},
    {"day": 4, "num_patients": 34,  "seed": 400},
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_env(num_patients: int, seed: int) -> SchedulingEnvironment:
    data = generate_random_data(
        num_patients=num_patients,
        max_ops=DEFAULT_MAX_OPS,
        skills=DEFAULT_SKILLS,
        task_probability=0.90,
        seed=seed,
    )
    return SchedulingEnvironment(data, DEFAULT_SKILLS, num_patients, DEFAULT_MAX_OPS)


def _run(env, agents_config, mode, use_ql, iters=ITERATIONS):
    """Lance un système, renvoie le meilleur makespan."""
    mas = MultiAgentSystem(env, mode=mode, use_qlearning=use_ql)
    for atype, aid, params in agents_config:
        mas.add_agent(atype, aid, **params)
    best = mas.run(n_iterations=iters, verbose=False)
    return best.fitness if best else float("inf"), mas

# ── Configurations d'agents ──────────────────────────────────────────────────

AG  = ("genetic", "AG",   {"population_size": 15})
TAB = ("tabu",    "Tabu", {"tabu_tenure": 10})
RS  = ("sa",      "RS",   {"initial_temp": 100})


def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABLEAU 1
# ══════════════════════════════════════════════════════════════════════════════

def run_table1():
    print("\n" + "=" * 100)
    print("  TABLEAU 1 : Sans Collaboration")
    print("=" * 100)

    header = (
        f"{'Jour':<5} | {'Patients':<9} | "
        f"{'AG':<6} {'Tabou':<6} {'RS':<6} | "
        f"{'Agent_AG':<9} {'Agent_Tabou':<12} {'Agent_RS':<9}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for day_cfg in DAYS:
        d = day_cfg["day"]
        n = day_cfg["num_patients"]
        env = _make_env(n, day_cfg["seed"])

        # Métaheuristiques sans apprentissage (agent seul, pas de QL)
        ag_val,  _  = _run(env, [AG],  CollaborationMode.ENEMIES, False)
        tab_val, _  = _run(env, [TAB], CollaborationMode.ENEMIES, False)
        rs_val,  _  = _run(env, [RS],  CollaborationMode.ENEMIES, False)

        # Métaheuristiques avec apprentissage (agent seul + QL)
        aag_val,  _  = _run(env, [AG],  CollaborationMode.ENEMIES, True)
        atab_val, _  = _run(env, [TAB], CollaborationMode.ENEMIES, True)
        ars_val,  _  = _run(env, [RS],  CollaborationMode.ENEMIES, True)

        row = {
            "day": d, "patients": n,
            "AG": ag_val, "Tabou": tab_val, "RS": rs_val,
            "Agent_AG": aag_val, "Agent_Tabou": atab_val, "Agent_RS": ars_val,
        }
        results.append(row)

        print(
            f"J{d:<4} | {n:<9} | "
            f"{ag_val:<6.0f} {tab_val:<6.0f} {rs_val:<6.0f} | "
            f"{aag_val:<9.0f} {atab_val:<12.0f} {ars_val:<9.0f}"
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TABLEAU 2
# ══════════════════════════════════════════════════════════════════════════════

# Combinaisons de collaboration du diapo
COLLAB_COMBOS = {
    "AG_Tabou":    [AG, TAB],
    "AG_RS":       [AG, RS],
    "Tabou_RS":    [TAB, RS],
    "AG_Tabou_RS": [AG, TAB, RS],
}


def _run_table2_block(use_ql: bool, label: str):
    """Bloc du tableau 2 : avec ou sans apprentissage."""
    print(f"\n--- Tableau 2 – {label} ---")

    combo_names = list(COLLAB_COMBOS.keys())
    col_width = 12

    # En-tête
    hdr = f"{'Jour':<5} | {'Pat.':<5} | "
    hdr += "AMIS: "
    for c in combo_names:
        hdr += f"{c:<{col_width}}"
    hdr += " | ENNEMIS: "
    for c in combo_names:
        hdr += f"{c:<{col_width}}"
    print(hdr)
    print("-" * len(hdr))

    results = []

    for day_cfg in DAYS:
        d = day_cfg["day"]
        n = day_cfg["num_patients"]
        env = _make_env(n, day_cfg["seed"])

        row = {"day": d, "patients": n}
        line = f"J{d:<4} | {n:<5} | "

        # Amis
        line += "      "
        for cname, configs in COLLAB_COMBOS.items():
            val, _ = _run(env, configs, CollaborationMode.FRIENDS, use_ql)
            row[f"amis_{cname}"] = val
            line += f"{val:<{col_width}.0f}"

        line += " |          "

        # Ennemis
        for cname, configs in COLLAB_COMBOS.items():
            val, _ = _run(env, configs, CollaborationMode.ENEMIES, use_ql)
            row[f"ennemis_{cname}"] = val
            line += f"{val:<{col_width}.0f}"

        results.append(row)
        print(line)

    return results


def run_table2():
    print("\n" + "=" * 100)
    print("  TABLEAU 2 : Avec Collaboration (Amis / Ennemis)")
    print("=" * 100)

    res_no_ql = _run_table2_block(use_ql=False, label="Sans Apprentissage")
    res_ql    = _run_table2_block(use_ql=True,  label="Avec Apprentissage")
    return res_no_ql, res_ql


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print("   BENCHMARK COMPLET – Tableaux 1 & 2 du diaporama")
    print("=" * 100)
    setup_output_dir()

    t1 = run_table1()
    t2_noql, t2_ql = run_table2()

    print("\n" + "=" * 100)
    print("✅  Benchmark terminé.")
    print("=" * 100)

    return t1, t2_noql, t2_ql


if __name__ == "__main__":
    main()
