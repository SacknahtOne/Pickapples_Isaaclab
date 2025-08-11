# leisaac/tasks/pick_apples/pick_apples_env_cfg.py
#Erstellt von Alexander Leinz, Cedric Dezsö, Matthis Klee, Jakob Chmil
#Ostfalia Wolfenbüttel, Fakultät Maschinenbau
# =============================================================================
#Diese Config beinhaltet die gesamte ,,Bauanleitung'' für die Umgebung PickApples
#Sie beinhaltet alle Objekte(Roboter, Apfel, Tisch...), wo diese stehen, welche Sensoren usw.
#Außerdem: Welche Beobachtungen, welche Aktionen, Resets, Events usw.
# =============================================================================
#Zuletzt bearbeitet: 11.08.25. Alexander Leinz

from dataclasses import MISSING  # Platzhalter: Macht dann die use_teleop_device 
from typing import Any
import torch  

# --- IsaacLab-Basistypen und Helfer (Namen sind sprechend) -------------------
import isaaclab.sim as sim_utils  # Spawner für primitive Formen, Lichter, PhysX-Props
from isaaclab.envs import ManagerBasedRLEnvCfg  # Oberste Env-Konfigurationsklasse
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg as RecordTerm,  # Registriert Zustände/Aktionen fürs Mitschneiden
)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # Roboter/Assets/Objekte
from isaaclab.managers import (
    EventTermCfg as EventTerm,          # Term, der bei Events (z. B. reset) ausgeführt wird
    ObservationGroupCfg as ObsGroup,    # Gruppe von Observationen (z. B. „policy“)
    ObservationTermCfg as ObsTerm,      # Einzelne Observation (z. B. joint_pos)
    RewardTermCfg as RewTerm,           # (hier leer, weil IL) — trotzdem importierbar
    SceneEntityCfg,                     # Referenz auf ein Szenenobjekt (z. B. Kamera „wrist“)
    TerminationTermCfg as DoneTerm,     # Abbruchbedingung (z. B. time_out)
    ActionTermCfg as ActTerm,           # Aktionsterm (z. B. Gelenkinkremente / Greifer)
)
from isaaclab.sensors import TiledCameraCfg   # Sensor-Config (RGB-Kamera mit Kachel-Layout)
from isaaclab.scene import InteractiveSceneCfg  # Szenen-Konfiguration (Assets, Sensoren, …)
from isaaclab.utils import configclass          # Dekorator, macht aus Klassen „Config-Klassen“

# --- Projekt-spezifische Bausteine (kommen aus deinem Repo) -------------------
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG
#  ^ vorgefertigte Roboterkonfiguration (Kinematik, Gelenke, Limits, Default-Props, …)

from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
#  ^ Hilfsfunktionen für Teleoperation: binden die richtigen Action-Terme anhand Device

from . import mdp  # unsere MDP-Utilities: Beobachtungsfunktionen, Reset-Randomisierung, Dones


# =============================================================================
# SZENE: Roboter, Tisch, Früchte, Schüssel, Kameras, Licht
# =============================================================================
@configclass
class PickApplesSceneCfg(InteractiveSceneCfg):
    """
    *Scene*-Konfiguration für die Aufgabe „PickApples“.
    In dieser Klasse listen wir ALLE Dinge auf, die in der Welt existieren sollen:
      - statische Assets (Boden, Licht),
      - Roboter (Articulation),
      - Rigid Bodies (Tisch, Apfel, Orange, Schüssel),
      - Sensoren (Kameras).
    Jede Eigenschaft hier ist wieder eine *Cfg*-Instanz, die sagt: „Wo? Wie groß?
    Mit welchen Materialeigenschaften?“. Die Engine nutzt das dann beim Spawn.
    """

    # --- Bodenebene (einfaches unendliches GroundPlane) -----------------------
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Scene",                    # USD-Pfad, wo das Asset hängen soll
        spawn=sim_utils.GroundPlaneCfg(),            # vordefinierter Boden (Kollision+Rendering)
    )

    # --- Roboter: 90° um Z-Achse nach links gedreht, etwas näher an den Tisch --
    # Quaternion-Konvention hier: (w, x, y, z). 90° um Z → cos(45°)=0.7071; sin(45°)=0.7071
    # Dadurch „schaut“ der Roboter quer zum Welt-X — so wie „mit der gings“-Setup.
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot",                    # USD-Pfad des Roboters
        init_state=ArticulationCfg.InitialStateCfg(  # Startpose des Basis-Links (Base)
            pos=(0.10, 0.00, 0.50),                  # XYZ in Metern — leicht erhöht/versetzt
            rot=(0.7071, 0.0, 0.0, 0.7071),         # 90°-Yaw (linksherum)
        ),
    )
    
    # zu kopieren und nur einzelne Felder anzupassen (prim_path/init_state).

    # --- Tisch: Position/Groesse wie in deiner funktionierenden Variante -------
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.00, 0.25)),
        spawn=sim_utils.CuboidCfg(                    # Quader-Geometrie (einfacher Tischblock)
            size=(0.60, 0.40, 0.50),                 # (Länge X, Breite Y, Höhe Z) in Metern
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),     # PhysX-Körper-Defaults
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0), # schwer/stabil
            collision_props=sim_utils.CollisionPropertiesCfg(), # Standard-Kollisionsmesh
            visual_material=sim_utils.PreviewSurfaceCfg(        # Optik (PBR „Preview Surface“)
                diffuse_color=(0.60, 0.40, 0.20),               # Holzton
                metallic=0.0,
                roughness=0.90,                                 # eher rau (wenig Glanz)
            ),
        ),
    )

    # --- Apfel:  ---------------------------------------
    red_apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/RedApple",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.13, 0.52)),
        spawn=sim_utils.SphereCfg(                   # Kugel als einfache Apfel-Näherung
            radius=0.03,                             # 3 cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.10, 0.05),    # Rotton
                metallic=0.0,
                roughness=0.70,                      # etwas glatter als Tisch
            ),
        ),
    )

    # --- Orange:  ------------------------------------------------
    orange: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Orange",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.10, 0.52)),
        spawn=sim_utils.SphereCfg(
            radius=0.035,                            # 3.5 cm — minimal größer als Apfel
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.125),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.55, 0.0),      # Orangeton
                metallic=0.0,
                roughness=0.80,                      # matte Schale
            ),
        ),
    )

    # --- Schüssel: mittig vor dem Roboter --------------------------------------
    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Bowl",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.00, 0.51)),
        spawn=sim_utils.CylinderCfg(                 # Zylinder als einfache Schüssel
            radius=0.08,
            height=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.50),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.80, 0.80, 0.80),    # hellgrau
                metallic=0.30,
                roughness=0.40,                      # etwas glänzender
            ),
        ),
    )

    # --- Wrist-Kamera: an der Greiferhand; Bildrauschen macht mdp.image_with_noise
    # Hinweis: Das *Sensor-Spawn* (Optik, FoV) definieren wir hier; die eigentliche
    # Bildkorruption (Rauschen, Gamma etc.) passiert in der MDP-Funktion pro Frame.
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(             # Pose relativ zum Mount-Link (Greifer)
            pos=(-0.001, 0.10, -0.04),              # leicht versetzt (in Metern)
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),  # Quaternion (w,x,y,z)
            convention="ros",                        # ROS-Achsdefinition (nur Orientierungskonv.)
        ),
        data_types=["rgb"],                          # wir lesen RGB-Bilder raus
        spawn=sim_utils.PinholeCameraCfg(            # Lochkamera (fester FoV/Clip)
            focal_length=36.5,
            focus_distance=400.0,                    # Fokus (nur optisch relevant)
            horizontal_aperture=36.83,               # steuert FOV
            clipping_range=(0.01, 50.0),             # Near/Far (Meter)
            lock_camera=True,                        # keine Maus-Interaktion im Viewer
        ),
        width=640, height=480,                       # Auflösung
        update_period=1.0 / 30.0,                    # 30 FPS
    )

    # --- Front-Kamera: am Roboter-Base-Frame, nach vorne blickend ---------------
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.5, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640, height=480,
        update_period=1.0 / 30.0,
    )

    # --- Umgebungslicht: DomeLight (gleichmäßig, neutral) -----------------------
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(                # kuppelförmige Lichtquelle
            color=(0.75, 0.75, 0.75),                # neutral-grau
            intensity=3000.0,                        # Baseline; mdp-Reset jittert sie leicht
        ),
    )


# =============================================================================
# ACTIONS / EVENTS / OBSERVATIONS / REWARDS / TERMINATIONS
# =============================================================================

@configclass
class ActionsCfg:
    """
    Welche *Aktionsterme* die Policy setzen darf.
    - Wir lassen die Felder absichtlich als MISSING, weil wir sie in
      use_teleop_device() dynamisch befüllen (abhängig von Keyboard/SpaceMouse).
    - Das erspart uns zwei Konfigurationsvarianten.
    """
    arm_action: ActTerm = MISSING       # z. B. kartesische Inkremente, oder Joint-Targets …
    gripper_action: ActTerm = MISSING   # z. B. Greifer auf/zu


@configclass
class EventCfg:
    """
    *Events* sind Hooks, die in bestimmten Modi laufen, z. B. beim Reset.
    Wir hängen unsere Randomisierung aus mdp daran, damit pro Episode leichte
    Variationen eingestreut werden (driftfrei, realistisch).
    """
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,  # ruft intern reset_scene_with_randomization auf
        mode="reset",
    )


@configclass
class ObservationsCfg:
    """
    Welche Sensor-/Zustandsgrößen sieht die Policy?
    - Wir definieren eine Gruppe „policy“, damit der ObservationManager weiß,
      welche Terme zusammengehören (und in welcher Form sie geliefert werden).
    - Kamerabilder werden NICHT concateniert, damit unsere mdp.image_with_noise
      die einzelnen Streams sauber korruptieren kann.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # --- propriozeptive Terme (kommen aus isaaclab.envs.mdp.*) -------------
        joint_pos     = ObsTerm(func=mdp.joint_pos)
        joint_vel     = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions       = ObsTerm(func=mdp.last_action)  # nützlich für IL/Teleop

        # --- Kameras mit milder Bildkorruption (Parameter → mdp.image_with_noise)
        # WICHTIG: Der ObservationManager erwartet bei Sensor-Terms gewisse
        # Parameter-Namen. „obs_cache“ ist Pflicht (auch wenn wir ihn nicht nutzen).
        wrist = ObsTerm(
            func=mdp.image_with_noise,
            params={
                "obs_cache": None,                    # Pflichtparameter laut Manager-API
                "sensor_cfg": SceneEntityCfg("wrist"),# verweist auf Szene-Sensor „wrist“
                "data_type": "rgb",                   # wir lesen RGB
                "normalize": False,                   # uint8 (0..255) — mdp kümmert sich
                "noise_std": 0.01,                    # ~1% additive Störung
                "bias_std": 0.01,                     # kleiner Frame-Bias
            },
        )
        front = ObsTerm(
            func=mdp.image_with_noise,
            params={
                "obs_cache": None,
                "sensor_cfg": SceneEntityCfg("front"),
                "data_type": "rgb",
                "normalize": False,
                "noise_std": 0.01,
                "bias_std": 0.01,
            },
        )

        def __post_init__(self):
            """
            Wird nach der automatischen Felderzeugung aufgerufen.
            - enable_corruption=True: erlaubt MDP-Terms, selbst Korruption/Normalisierung
              durchzuführen (hier: mdp.image_with_noise).
            - concatenate_terms=False: Kamerabilder nicht zusammenkleben, sonst können
              Formate/Noise-Pipelines durcheinander geraten.
            """
            self.enable_corruption = True
            self.concatenate_terms = False

    # Die Gruppe „policy“ aktivieren
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """
    Für Imitation Learning (Teleoperation + Nachahmung) brauchen wir hier nichts.
    Trotzdem lassen wir die Klasse stehen (erweitern ist später leicht).
    """
    pass


@configclass
class TerminationsCfg: #wir haben eine terminations.py
    """
    Wann endet eine Episode?
    - time_out: Standard, endet nach max. Episodenlänge (siehe Env __post_init__).
    - success: Beispiel auskommentiert (kannst du später reaktivieren, wenn
      du das Ziel „Apfel in Schüssel“ wieder als Done verwenden willst).
    """

    # Beispiel-Erfolg (auskommentiert):
    # success = DoneTerm(
    #     func=mdp.apple_in_bowl,
    #     params={
    #         "apple_cfg": SceneEntityCfg("red_apple"),
    #         "bowl_cfg":  SceneEntityCfg("bowl"),
    #         "distance_threshold": 0.10,  # 10 cm
    #     },
    #     time_out=False,
    # )

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# =============================================================================
# UMWELT: Alles zusammenstecken (Scene + Manager-Konfigurationen)
# =============================================================================
@configclass
class PickApplesEnvCfg(ManagerBasedRLEnvCfg):
    """
    Oberste Env-Konfigurationsklasse.
    - Erbt von ManagerBasedRLEnvCfg → bringt Render-/PhysX-/Viewer-Parameter mit.
    - Wir stecken Scene, Observations, Actions, Events, Rewards, Terminations und
      Recorder zusammen.
    """

    # Szene: 1 Environment (duplizieren geht später über num_envs)
    scene: PickApplesSceneCfg = PickApplesSceneCfg(num_envs=1, env_spacing=8.0)
    #  ^ env_spacing: Abstand zwischen parallelen Envs (bei num_envs>1)

    # Manager-Konfigurationen
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    recorders: RecordTerm = RecordTerm()  # zeichnet Zustände/Aktionen auf (hilfreich für Demos)

    def __post_init__(self) -> None:
        """
        Nachinitialisierung: Viewer-Position, PhysX-Feinheiten, Episode-Länge …
        Hier kommen Dinge rein, die nicht direkt Teil der Scene sind.
        """
        super().__post_init__()

        # --- Viewer (Kamera im Isaac Sim-Viewport) -----------------------------
        self.viewer.eye = (-0.8, -1.2, 1.5)   # Kamera-Position
        self.viewer.lookat = (0.4, 0.0, 0.5)  # Blickpunkt

        # --- Physik / Simulation ------------------------------------------------
        self.decimation = 1                 # 1 = jede Physik-Iteration ist ein Env-Step
        self.episode_length_s = 30.0        # Time-Out (siehe TerminationsCfg.time_out)
        # Stabilität/Feinheiten (konservative Defaults):
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True  # Glas/halbtransparente Materialien

    # =============================================================================
    # Teleoperation (Keyboard/SpaceMouse): Aktionen dynamisch „verdrahten“
    # =============================================================================
    def use_teleop_device(self, teleop_device) -> None:
        """
        Wird vom Teleop-Skript aufgerufen. Wir befüllen die Action-Konfiguration
        passend zum Eingabegerät (Keyboard, Spacemouse, …).
        - init_action_cfg() kommt aus deinem Projekt und weiß, welche ActTerms
          zu setzen sind (arm_action/gripper_action).
        - Mini-Spezialfall: Beim Keyboard schalten wir die Gravitation für den
          Roboter-Basislink aus, damit die Base nicht ungewollt kippt/rutscht.
        """
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device == "keyboard":
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        """
        Optionaler Hook, um Roh-Eingaben (Tasten/Axis) in ein Policy-kompatibles
        Aktionsformat zu bringen (Skalierung, Mapping, Clipping …).
        - Rückgabe: Torch-Tensor (wird direkt als Action in die Env eingespielt).
        """
        return preprocess_device_action(action, teleop_device)
