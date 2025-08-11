# leisaac/tasks/pick_apples/__init__.py
#Erstellt von Alexander Leinz, Cedric Dezsö, Matthis Klee, Jakob Chmil
#Ostfalia Wolfenbüttel, Fakultät Maschinenbau
# =============================================================================
#Diese init sorgt dafür dass Python den ordner Pickapples als Modul erkennt und ihn in
#andere Dateien per import einbinden kann
#Importiert Umgebungskonfiguration über registery-Mechanismus damit man die Datei starten kann
#bzw. damit sie als Einstiegspunkt in der Isaac-Lab Struktur auffindbar ist
# =============================================================================
#Zuletzt bearbeitet: 11.08.25. Alexander Leinz

import gymnasium as gym

gym.register(
    id='LeIsaac-SO101-PickApples-v0',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_apples_env_cfg:PickApplesEnvCfg",
    },
)
