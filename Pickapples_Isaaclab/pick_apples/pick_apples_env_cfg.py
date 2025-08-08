#Verfasser: Alexander Leinz, Cedric Dezsö, Jakob Chmil, Matthis Klee 08.08.25
#in Zusammenarbeit mit der Ostfalia Fakultät Maschinenbau
#Zuletzt bearbeitet: 08.08.25
#Als Hilfestellung und zur Kommentierung wurde Claude.ai verwendet

#########################################################################
#Rechtfertigungen, warum wir etwas (nicht) gemacht haben sind eingerahmt#
#########################################################################

################################################################################################
# Import-Statements: Hier holen wir uns fertige Code-Bausteine von anderen Dateien
# "import" ist wie "hole mir das Werkzeug X aus der Werkzeugkiste Y"
import torch  # PyTorch ist eine Bibliothek für maschinelles Lernen und Tensor-Operationen
from dataclasses import MISSING  # MISSING ist ein spezieller Platzhalter der sagt "hier fehlt noch was"
from typing import Any  # "Any" bedeutet "beliebiger Datentyp" - für Type Hints in Python

# Isaac Lab spezifische Imports - das ist die Roboter-Simulations-Software
import isaaclab.sim as sim_utils  # Simulations-Utilities (Hilfsfunktionen für die Simulation)
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm  # Zum Aufzeichnen von Aktionen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # Konfigurationen für verschiedene Objekt-Typen
from isaaclab.envs import ManagerBasedRLEnvCfg  # Basis-Klasse für Reinforcement Learning Umgebungen
from isaaclab.managers import EventTermCfg as EventTerm  # Für Events (Ereignisse) in der Simulation
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # Gruppiert Beobachtungen zusammen
from isaaclab.managers import ObservationTermCfg as ObsTerm  # Einzelne Beobachtungen (was der Roboter "sieht")
from isaaclab.managers import RewardTermCfg as RewTerm  # Belohnungen (brauchen wir nicht, aber muss da sein)
from isaaclab.managers import SceneEntityCfg  # Konfiguration für Objekte in der Szene
from isaaclab.managers import TerminationTermCfg as DoneTerm  # Wann ist eine Episode zu Ende?
from isaaclab.sensors import TiledCameraCfg  # Konfiguration für Kameras
from isaaclab.scene import InteractiveSceneCfg  # Basis-Klasse für interaktive Szenen
from isaaclab.utils import configclass  # Spezieller Decorator der aus Klassen Konfigurations-Klassen macht
from isaaclab.utils.noise import NoiseCfg, NoiseModelWithAdditiveBiasCfg  # Für Rauschen/Störungen in Sensoren

# Eigene Imports aus deinem Projekt
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG  # Die Konfiguration deines SO-101 Roboters
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action  # Für Steuergeräte (Keyboard, etc.)

# Importiere MDP (Markov Decision Process) Funktionen - das sind vordefinierte Funktionen für die Umgebung
from . import mdp  # Der Punkt bedeutet "aus dem gleichen Ordner"

import math #Für pi/2 --> 90°
# @configclass ist ein "Decorator" - das ist wie ein Stempel der sagt:
# "Diese Klasse ist eine spezielle Konfigurations-Klasse"
@configclass
class PickApplesSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the pick apples task with domain randomization.
    
    Diese Klasse definiert ALLES was in deiner Simulations-Welt existiert:
    - Wo ist der Boden?
    - Wo steht der Roboter?
    - Wo sind die Äpfel?
    - Wo ist die Schüssel?
    - Welche Kameras gibt es?
    
    Eine Klasse ist wie ein Bauplan. InteractiveSceneCfg ist die Eltern-Klasse (Vorlage).
    """

    # Der Boden auf dem alles steht - ohne den würde alles ins Leere fallen!
    # AssetBaseCfg ist die allgemeinste Form eines "Dings" in der Simulation
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Scene",  # Der "Pfad" wo das Objekt in der Simulations-Hierarchie liegt (wie ein Dateipfad)
        spawn=sim_utils.GroundPlaneCfg()  # GroundPlaneCfg() erzeugt einen flachen Boden
    )

    # Der Roboter - das Herzstück deiner Simulation!
    # ArticulationCfg ist für Objekte mit Gelenken (wie Roboter)
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(  # Wir nehmen die SO101 Basis-Config und ändern nur bestimmte Teile
        prim_path="/World/Robot",  # Wo der Roboter in der Hierarchie liegt
        init_state=ArticulationCfg.InitialStateCfg(  # InitialStateCfg definiert die Startposition
            pos=(0.1, 0.0, 0.5),  # Position in 3D: (x=10cm vor, y=0 seitlich, z=50cm hoch) - Roboter steht AUF dem Tisch!
            rot=(0.7071, 0.0, 0.0, 0.7071)  # Rotation als Quaternion (1,0,0,0 = keine Drehung) - Mathe für 3D-Drehungen #Wir müssen 90° um Z drehen also 0.7071, 0.0, 0.0, 0.7071
        )
    )

    # Der rote Apfel - das soll der Roboter greifen
    # RigidObjectCfg ist für starre Objekte (keine Gelenke, aber Physik-Simulation)
    red_apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/RedApple",  # Pfad in der Hierarchie
        init_state=RigidObjectCfg.InitialStateCfg(  # Startposition des Apfels
            pos=(0.3, 0.13, 0.52),  # Position: 25cm vor Roboter, 10cm rechts, 52cm hoch (auf dem Tisch)
        ),
        spawn=sim_utils.SphereCfg(  # SphereCfg = Kugel-Form (Äpfel sind rund!)
            radius=0.03,  # Radius = 3cm (ein kleiner Apfel)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Standard Physik-Eigenschaften (Schwerkraft wirkt, kann sich bewegen)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # Gewicht: 100 Gramm (realistisch für einen kleinen Apfel)
            collision_props=sim_utils.CollisionPropertiesCfg(),  # Kollisions-Eigenschaften (kann andere Objekte berühren)
            visual_material=sim_utils.PreviewSurfaceCfg(  # Wie sieht der Apfel aus?
                diffuse_color=(1.0, 0.0, 0.0),  # RGB Farbe: (Rot=100%, Grün=0%, Blau=0%) = ROT!
                metallic=0.0,  # 0 = nicht metallisch (Äpfel glänzen nicht wie Metall)
                roughness=0.7,  # 0.7 = ziemlich rau (matter Apfel, nicht hochglanz)
            ),
        ),
    )

    # Die Orange - zweites Objekt zum Greifen (für Abwechslung)
    # Sehr ähnlich zum Apfel, nur andere Werte
    orange: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Orange",  # Eigener Pfad für die Orange
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, -0.1, 0.52),  # Links vom Roboter (y=-0.1 bedeutet 10cm links)
        ),
        spawn=sim_utils.SphereCfg(  # Auch eine Kugel
            radius=0.035,  # Etwas größer als der Apfel (3.5cm Radius)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Standard Physik
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),  # 120 Gramm (etwas schwerer)
            collision_props=sim_utils.CollisionPropertiesCfg(),  # Kann kollidieren
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),  # RGB: Rot=100%, Grün=50%, Blau=0% = ORANGE!
                metallic=0.0,  # Nicht metallisch
                roughness=0.7,  # Auch matt
            ),
        ),
    )

    # Die Schüssel - hier soll der Roboter die Früchte reinlegen
    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Bowl",  # Pfad der Schüssel
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.51),  # 30cm vor dem Roboter, mittig, auf dem Tisch
        ),
        spawn=sim_utils.CylinderCfg(  # CylinderCfg = Zylinder-Form (Schüsseln sind rund und flach)
            radius=0.08,  # 8cm Radius (16cm Durchmesser)
            height=0.04,  # 4cm hoch (flache Schüssel)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Standard Physik
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),  # 500 Gramm (schwere Schüssel, bewegt sich nicht leicht)
            collision_props=sim_utils.CollisionPropertiesCfg(),  # Kann Kollisionen haben
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8),  # Hellgrau (wie Keramik)
                metallic=0.3,  # Ein bisschen glänzend
                roughness=0.4,  # Relativ glatt
            ),
        ),
    )

    # Der Tisch - darauf steht alles
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Table",  # Pfad des Tisches
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.25),  # Position: 30cm vor Ursprung, Höhe 25cm (das ist die MITTE des Tisches!)
        ),
        spawn=sim_utils.CuboidCfg(  # CuboidCfg = Quader/Box-Form
            size=(0.6, 0.4, 0.5),  # Größe: 60cm lang, 40cm breit, 50cm hoch
            # Da der Tisch bei z=0.25 zentriert ist und 0.5m hoch ist:
            # Unterkante bei 0.0, Oberkante bei 0.5 - perfekt!
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Standard Physik
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),  # 100kg! Sehr schwer damit er stabil steht
            collision_props=sim_utils.CollisionPropertiesCfg(),  # Kann Kollisionen haben
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.4, 0.2),  # Braun wie Holz
                metallic=0.0,  # Kein Metall
                roughness=0.9,  # Sehr rau (Holzstruktur)
            ),
        ),
    )

    # Handgelenk-Kamera - direkt am Greifer montiert
    # TiledCameraCfg ist speziell für Kameras in Isaac Sim
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/gripper/wrist_camera",  # Pfad: An den Greifer des Roboters gehängt
        offset=TiledCameraCfg.OffsetCfg(  # Versatz von der Montageposition
            pos=(-0.001, 0.1, -0.04),  # Kleine Verschiebung für optimale Sicht
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),  # Rotation als Quaternion (komplizierte Mathe für 3D-Drehung)
            convention="ros"  # ROS = Robot Operating System Konvention für Koordinaten
        ),
        data_types=["rgb"],  # Nur RGB-Bilder (keine Tiefe, kein Infrarot)
        spawn=sim_utils.PinholeCameraCfg(  # Lochkamera-Modell (Standard Kamera-Typ)
            focal_length=36.5,  # Brennweite in mm (wie Zoom-Level)
            focus_distance=400.0,  # Fokus-Distanz in mm
            horizontal_aperture=36.83,  # Sensor-Breite in mm
            clipping_range=(0.01, 50.0),  # Rendert nur Objekte zwischen 1cm und 50m Entfernung
            lock_camera=True  # Kamera ist fest montiert, wackelt nicht
        ),
        width=640,  # Bildbreite in Pixel
        height=480,  # Bildhöhe in Pixel
        update_period=1 / 30.0,  # 30 FPS (Bilder pro Sekunde)
    )

    # Front-Kamera - zeigt die ganze Szene von vorne
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/base/front_camera",  # An der Roboter-Basis montiert
        offset=TiledCameraCfg.OffsetCfg(  # Versatz für gute Übersicht
            pos=(0.0, -0.5, 0.6),  # 50cm nach hinten, 60cm hoch
            rot=(0.1650476, -0.9862856, 0.0, 0.0),  # Leicht nach unten geneigt
            convention="ros"  # ROS Konvention
        ),
        data_types=["rgb"],  # RGB Bilder
        spawn=sim_utils.PinholeCameraCfg(  # Kamera-Eigenschaften
            focal_length=28.7,  # Weitwinkel (kleinere Brennweite = größerer Blickwinkel)
            focus_distance=400.0,  # Fokus
            horizontal_aperture=38.11,  # Sensor-Größe
            clipping_range=(0.01, 50.0),  # Render-Bereich
            lock_camera=True  # Fest montiert
        ),
        width=640,  # Gleiche Auflösung wie Handkamera
        height=480,  # Standard VGA Auflösung
        update_period=1 / 30.0,  # 30 FPS
    )

    # Beleuchtung - ohne Licht sieht man nichts!
    light = AssetBaseCfg(
        prim_path="/World/Light",  # Pfad des Lichts
        spawn=sim_utils.DomeLightCfg(  # Dome Light = Himmelskuppel-Beleuchtung (von überall)
            color=(0.75, 0.75, 0.75),  # Leicht graues Licht (nicht ganz weiß)
            intensity=3000.0  # Lichtstärke (3000 ist ziemlich hell)
        ),
    )


# Konfiguration für die Aktionen (was kann der Roboter tun?)
@configclass
class ActionsCfg:
    """Configuration for the actions.
    
    Aktionen sind die Befehle die wir dem Roboter geben können.
    Z.B. "Bewege Gelenk 1 um 10 Grad" oder "Öffne den Greifer"
    """
    # MISSING bedeutet: "Wird später ausgefüllt" - das macht die init_action_cfg Funktion
    arm_action: mdp.ActionTermCfg = MISSING  # Steuerung für die Arm-Gelenke
    gripper_action: mdp.ActionTermCfg = MISSING  # Steuerung für den Greifer (auf/zu)


# Konfiguration für Events (Ereignisse in der Simulation)
@configclass
class EventCfg:
    """Configuration for the events with domain randomization.
    
    Events sind Dinge die passieren, z.B.:
    - Reset der Szene
    - Zufällige Änderungen (Domain Randomization)
    - Störungen
    """
    
    # Reset Event - setzt alles auf Anfangsposition zurück
    # EventTerm ist ein einzelnes Event
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,  # Diese Funktion wird aufgerufen
        mode="reset"  # Wann? Bei jedem Reset (neue Episode)
    )
    
    # HINWEIS: Domain Randomization ist wichtig für sim-to-real!
    # Da die mdp Funktionen nicht existieren, machen wir die Randomisierung
    # direkt in der Scene-Config (siehe unten bei Apfel/Orange Definitionen)
    # 
    # Alternative: Erstelle eigene Randomisierungs-Funktionen in mdp/__init__.py
    # oder nutze die Standard Isaac Lab mdp Funktionen aus:
    # from isaaclab.envs import mdp as base_mdp
    #
    ################################################################
    #Wir haben das versucht, wir vermuten: wir kollidieren immer wieder mit der mdp der pick oranges, an der wir uns orientiert haben. Für dieses Projekt nehmen wir einfach die Python random Funktion.
    #Es wird nicht alles randomisiert. Licht bspw ist immer gleich, könnte aber auch ganz einfach mit random gemacht werden. Für kleine Projekte wie hier ist das ausreichend, für größere Projekte
    #sollte man sich mit mdp auseinander setzen, da es auch viel einfacher ist und realitätsnaher. Natürlich nur, wenn man es zum Laufen bekommt... Wir sind daran gescheitert.
    #################################################################
    # Für jetzt lassen wir die Events leer und nutzen die eingebaute
    # Varianz durch Physik-Simulation und Sensor-Rauschen


# Konfiguration für Beobachtungen (was "sieht" der Roboter?)
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    
    Beobachtungen sind alle Informationen die der Roboter/die KI bekommt:
    - Gelenkpositionen
    - Geschwindigkeiten
    - Kamerabilder
    - etc.
    """

    # Verschachtelte Klasse (Klasse in einer Klasse)
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        
        Policy = Die KI die den Roboter steuert
        Diese Gruppe definiert ALLE Inputs die die KI bekommt
        """
        
        # Gelenkposition - wo ist jedes Gelenk gerade?
        joint_pos = ObsTerm(func=mdp.joint_pos)
        
        # Gelenkgeschwindigkeit - wie schnell bewegt sich jedes Gelenk?
        joint_vel = ObsTerm(func=mdp.joint_vel)
        
        # Relative Gelenkposition - Differenz zur Ruheposition
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        
        # Relative Gelenkgeschwindigkeit
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        
        # Die letzte Aktion - was wurde als letztes befohlen?
        actions = ObsTerm(func=mdp.last_action)
        
        # Handgelenk-Kamera Bild
        wrist = ObsTerm(
            func=mdp.image,  # Funktion zum Bild holen
            params={  # Parameter (zusätzliche Einstellungen)
                "sensor_cfg": SceneEntityCfg("wrist"),  # Welche Kamera? Die "wrist" Kamera
                "data_type": "rgb",  # RGB Farbbild
                "normalize": False  # Nicht normalisieren (Pixelwerte bleiben 0-255)
            }
        )
        
        # Front-Kamera Bild
        front = ObsTerm(
            func=mdp.image,  # Gleiche Funktion
            params={  # Aber andere Kamera
                "sensor_cfg": SceneEntityCfg("front"),  # Die "front" Kamera
                "data_type": "rgb",  # RGB
                "normalize": False  # Keine Normalisierung
            }
        )

        # __post_init__ wird automatisch nach der Initialisierung aufgerufen
        def __post_init__(self):
            self.enable_corruption = True  # Erlaube Rauschen/Störungen
            self.concatenate_terms = False  # Nicht alles zu einem großen Vektor zusammenfügen

    # Erstelle eine Instanz der PolicyCfg Klasse
    policy: PolicyCfg = PolicyCfg()


# Konfiguration für Belohnungen (Rewards)
@configclass
class RewardsCfg:
    """Empty rewards config - not needed for imitation learning.
    
    Bei Imitation Learning lernt der Roboter durch Nachmachen,
    nicht durch Belohnungen. Deswegen ist diese Klasse leer.
    
    Bei Reinforcement Learning würden hier Belohnungen definiert:
    +10 Punkte für "Apfel gegriffen"
    -5 Punkte für "Apfel fallen gelassen"
    etc.
    """
    pass  # "pass" bedeutet: Diese Klasse ist absichtlich leer

#########################################################################################################################################
#Die Reward Funktion führt zu Fehlern und wir brauche sie nicht für Imitation Learning. Wir vermuten es hängt mit dem MDP Error zusammen#
#########################################################################################################################################

# Konfiguration für Beendigungen (wann ist eine Episode vorbei?)
@configclass
class TerminationsCfg:
    """Configuration for the terminations.
    
    Terminations beenden eine Episode, z.B.:
    - Zeitlimit erreicht
    - Aufgabe erfolgreich
    - Kritischer Fehler
    """
    # Zeitlimit - Episode endet nach bestimmter Zeit
    time_out = DoneTerm(
        func=mdp.time_out,  # Funktion die prüft ob Zeit um ist
        time_out=True  # Markiere dies als Timeout (nicht als Fehler)
    )


# Die Haupt-Konfigurations-Klasse - bringt alles zusammen!
@configclass
class PickApplesEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick apples environment for imitation learning.
    
    Das ist die HAUPTKLASSE die alles zusammenführt:
    - Die Szene (Objekte, Roboter, etc.)
    - Die Beobachtungen (was sieht der Roboter)
    - Die Aktionen (was kann er tun)
    - Events (was passiert)
    - Rewards (Belohnungen - hier leer)
    - Terminations (wann ist Ende)
    
    ManagerBasedRLEnvCfg ist die Basis-Klasse von Isaac Lab
    """

    # Alle Komponenten werden hier zusammengeführt:
    scene: PickApplesSceneCfg = PickApplesSceneCfg(
        num_envs=1,  # Anzahl paralleler Umgebungen (1 = nur eine)
        env_spacing=8.0  # Abstand zwischen Umgebungen (irrelevant bei num_envs=1)
    )

    # Verbinde alle Konfigurationen
    observations: ObservationsCfg = ObservationsCfg()  # Was der Roboter sieht
    actions: ActionsCfg = ActionsCfg()  # Was der Roboter tun kann
    events: EventCfg = EventCfg()  # Was passiert (Events)
    rewards: RewardsCfg = RewardsCfg()  # Belohnungen (leer bei Imitation Learning)
    terminations: TerminationsCfg = TerminationsCfg()  # Wann endet eine Episode

    # Recorder für das Aufzeichnen von Demonstrationen
    recorders: RecordTerm = RecordTerm()  # Speichert alles in HDF5 Dateien

    # __post_init__ wird automatisch nach der Initialisierung aufgerufen
    def __post_init__(self) -> None:
        """Diese Funktion wird automatisch aufgerufen nachdem die Klasse erstellt wurde.
        
        Hier können wir zusätzliche Einstellungen machen die nicht direkt
        als Klassen-Variablen definiert werden können.
        
        -> None bedeutet: Diese Funktion gibt nichts zurück (void in anderen Sprachen)
        """
        # Rufe die __post_init__ der Elternklasse auf (wichtig!)
        super().__post_init__()  # super() = Die Elternklasse (ManagerBasedRLEnvCfg)

        # Simulations-Einstellungen
        self.decimation = 1  # Jeden wievielten Simulations-Schritt nutzen (1 = alle)
        self.episode_length_s = 50.0  # Episode dauert 50 Sekunden
        
        # Kamera-Position für den Viewer (das Fenster wo du zuschaust)
        self.viewer.eye = (-0.8, -1.2, 1.5)  # Position der "Kamera" im 3D Raum
        self.viewer.lookat = (0.4, 0.0, 0.5)  # Wohin schaut die Kamera
        
        # Physik-Einstellungen (PhysX ist die Physik-Engine)
        self.sim.physx.bounce_threshold_velocity = 0.01  # Ab welcher Geschwindigkeit hüpfen Objekte
        self.sim.physx.friction_correlation_distance = 0.00625  # Technischer Reibungs-Parameter
        self.sim.render.enable_translucency = True  # Erlaube durchsichtige Objekte
        
        ###########################################
        # MANUELLE DOMAIN RANDOMIZATION (ohne mdp)########
        # Wir variieren die Werte bei jedem Reset manuell#
        ##################################################
        import random
        
        # Randomisiere Apfel-Eigenschaften
        if hasattr(self.scene, 'red_apple'): #IF: Falls das Attribut red_apple im Objekt self.scene existiert, führe den folgenden code aus:
            # Variiere Masse (80-120g)
            base_mass = 0.1
            self.scene.red_apple.spawn.mass_props.mass = base_mass * random.uniform(0.8, 1.2)
            
            # Variiere Größe leicht (±10%)
            base_radius = 0.03
            self.scene.red_apple.spawn.radius = base_radius * random.uniform(0.9, 1.1)
            
            # Variiere Farbe leicht (verschiedene Rottöne)
            red_variation = random.uniform(0.8, 1.0)
            green_variation = random.uniform(0.0, 0.2)
            self.scene.red_apple.spawn.visual_material.diffuse_color = (red_variation, green_variation, 0.05)
        
        # Randomisiere Orange-Eigenschaften
        if hasattr(self.scene, 'orange'):
            # Variiere Masse (100-150g)
            base_mass = 0.125
            self.scene.orange.spawn.mass_props.mass = base_mass * random.uniform(0.8, 1.2)
            
            # Variiere Größe leicht (±10%)
            base_radius = 0.035
            self.scene.orange.spawn.radius = base_radius * random.uniform(0.9, 1.1)
            
            # Variiere Orange-Farbe
            green_variation = random.uniform(0.5, 0.65)
            self.scene.orange.spawn.visual_material.diffuse_color = (1.0, green_variation, 0.0)# Ab welcher Geschwindigkeit hüpfen Objekte
        self.sim.physx.friction_correlation_distance = 0.00625  # Technischer Reibungs-Parameter
        self.sim.render.enable_translucency = True  # Erlaube durchsichtige Objekte

        if hasattr(self.scene, "table"):
            # Rauigkeit: Holz kann matt bis halbglänzend sein (z.B. 0.7 bis 0.95)
            self.scene.table.spawn.visual_material.roughness = random.uniform(0.7, 0.95)
            # Farbe: z.B. verschiedene Brauntöne (RGB)
            brown = random.uniform(0.5, 0.7)
            greenish = random.uniform(0.35, 0.5)
            redish = random.uniform(0.18, 0.28)
            self.scene.table.spawn.visual_material.diffuse_color = (brown, greenish, redish)

        if hasattr(self.scene, "bowl"):
            # Rauigkeit: z.B. 0.2 (glatt) bis 0.7 (matt)
            self.scene.bowl.spawn.visual_material.roughness = random.uniform(0.2, 0.7)
            # Farbe: Grautöne oder auch dezent andere Töne
            color_val = random.uniform(0.6, 0.95)
            self.scene.bowl.spawn.visual_material.diffuse_color = (color_val, color_val, color_val)
            # Metallischer Glanz: z.B. 0 (Keramik) bis 0.6 (Metallic-Look)
            self.scene.bowl.spawn.visual_material.metallic = random.uniform(0.0, 0.6)
            # Größe: Maximal 10% Variation
            base_radius = 0.08
            base_height = 0.04
            self.scene.bowl.spawn.radius = base_radius * random.uniform(0.9, 1.1)
            self.scene.bowl.spawn.height = base_height * random.uniform(0.9, 1.1)


        


    def use_teleop_device(self, teleop_device) -> None:
        """Konfiguriere die Umgebung für ein bestimmtes Steuergerät.
        
        Args:
            teleop_device: String wie "keyboard", "spacemouse", etc.
            
        Diese Funktion wird aufgerufen wenn du z.B. --teleop_device=keyboard verwendest
        """
        # Initialisiere die Actions für das gewählte Gerät
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        
        # Spezial-Einstellung für Keyboard
        if teleop_device == "keyboard":  # Wenn Keyboard verwendet wird
            # Schalte Schwerkraft für Roboter aus (einfacher zu steuern)
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        """Wandle Geräte-Input in Roboter-Aktionen um.
        
        Args:
            action: Dictionary mit den rohen Eingaben vom Gerät
            teleop_device: Welches Gerät wird verwendet
            
        Returns:
            torch.Tensor: Die verarbeiteten Aktionen als Tensor (spezielle Array-Form)
            
        Diese Funktion übersetzt z.B. Tastatureingaben in Gelenkbewegungen
        """
        return preprocess_device_action(action, teleop_device)
