# leisaac/tasks/pick_apples/mdp/__init__.py
#Erstellt von Alexander Leinz, Cedric Dezsö, Matthis Klee, Jakob Chmil
#Ostfalia Wolfenbüttel, Fakultät Maschinenbau
# =============================================================================
# Diese init bündelt alle wichtigen Hilfsfunktionen und Konstanten für die MDP
#,,Markov Decision Process'' -Logik (Jittern)
# =============================================================================
#Zuletzt bearbeitet: 11.08.25. Alexander Leinz

from __future__ import annotations  

import math
import random
from typing import TYPE_CHECKING

import torch  

try:
    from isaaclab.envs.mdp import * 
except ImportError:
    pass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  


# =============================================================================
# 1) Beobachtungen: Rohbild lesen + realistische Bildkorruption (Rauschen etc.)
# =============================================================================

def last_action(env):
    """
    Liefert die zuletzt gesetzte Aktion (nützlich bei Imitation/Teleop).
    - env.action_manager.action ist i. d. R. ein Tensor der letzten Befehle.
    """
    return env.action_manager.action


def image(env, obs_cache, sensor_cfg, data_type: str = "rgb", normalize: bool = False):
    """
    Basishook zum Lesen von Sensordaten (z. B. Kamerabild) aus der Szene.

    Parameter (vom ObservationManager vorgegeben):
      - env:         die Umgebung (hat .scene, .managers, …)
      - obs_cache:   Cache-Objekt (hier ungenutzt, aber *Pflichtparameter*)
      - sensor_cfg:  beschreibt *welcher* Sensor gelesen wird; wir nutzen den Namen
      - data_type:   "rgb" (Standard). Andere Typen wären z. B. "depth".
      - normalize:   False → uint8 [0..255], True → float [0..1]

    Rückgabe:
      - Tensor oder ndarray mit den Sensordaten (wir wandeln später bei Bedarf).
    """
    return env.scene.sensors[sensor_cfg.name].data.output[data_type]


# ---------------------------
# Hilfsfunktionen Bildpipeline
# ---------------------------

def _apply_gamma(x_float01: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Wendet Gamma-Korrektur auf Werte in [0..1] an.
    - gamma > 1 → dunkler, gamma < 1 → heller (nichtlinear, realistischer als nur Add/Mul).
    """
    return torch.clamp(x_float01, 0.0, 1.0).pow(1.0 / gamma)


def _apply_wb(x_float01: torch.Tensor, rgb_gain: torch.Tensor) -> torch.Tensor:
    """
    Einfacher Weißabgleich: skaliert die 3 Farbkanäle (R,G,B) separat.
    - rgb_gain-Form: (N,1,1,3); Mittelwert ~1, damit Helligkeit im Schnitt stabil bleibt.
    """
    return torch.clamp(x_float01 * rgb_gain, 0.0, 1.0)


def _read_light_intensity(default: float = 3000.0) -> float:
    """
    Liest die aktuelle DomeLight-Intensität vom USD-Prim '/World/Light'.
    Zweck: Belichtung minimal an Lichtstärke koppeln (realistischer).
    Fallback: 'default', falls das Attribut nicht verfügbar ist.
    """
    try:
        from omni.isaac.core.utils.prims import get_prim_at_path
        prim = get_prim_at_path("/World/Light")
        if prim:
            attr = prim.GetAttribute("intensity")
            if attr and attr.HasAuthoredValueOpinion():
                val = attr.Get()
                if isinstance(val, (float, int)):
                    return float(val)
    except Exception:
        pass
    return default


def image_with_noise(
    env,
    obs_cache,
    sensor_cfg,
    data_type: str = "rgb",
    normalize: bool = False,
    # --- Rausch-Parameter (Std-Abw. in Bildskala) ---
    noise_std: float = 0.01,  # Pixelrauschen (float: ~1%; uint8: ~2.55 Graylevels)
    bias_std: float = 0.01,   # Frame-Bias (globaler Offset pro Frame)
    # --- Photometrie linear ---
    brightness_jitter: float = 0.02,  # Helligkeits-Offset ±2%
    contrast_jitter: float = 0.02,    # Kontrast-Skalierung ±2%
    # --- Nichtlinear / Farb-Balance ---
    gamma_jitter: float = 0.05,       # Gamma ca. [0.95..1.05]
    cct_jitter: float = 0.03,         # Weißabgleich ±3 % pro Kanal (normiert auf Mittel=1)
    # --- Artefakte ---
    dropout_prob: float = 0.0002,     # seltene tote Pixel
    hotpixel_prob: float = 0.00005,   # sehr seltene heiße Pixel
    # --- Belichtung an Licht koppeln (sehr mild) ---
    exposure_base: float = 3000.0,          # Basisintensität des DomeLights
    exposure_brightness_gain: float = 0.2,  # 10% Licht → ~2% Helligkeits-Offset
    exposure_gamma_gain: float = 0.10,      # 10% Licht → ~1% Gamma-Shift
):
    """
    Realistische Bildkorruption in *kleinen*, sichtbaren Dosen (IL-freundlich).
    Pipeline:
      0) Belichtung aus Lichtintensität ableiten (exposure_factor)
      1) Frame-Bias (ein globaler Offset pro Frame)
      2) Pixelrauschen (pro Pixel)
      3) Brightness/Contrast + belichtungsabhängiger Offset
      4) Gamma + Weißabgleich (auf [0..1] normalisiert)
      5) selten Dead/Hot-Pixels
      6) Clamp + Rück-Cast auf Ursprungstyp (z. B. uint8)
    """
    # Rohbild holen (kann Tensor oder numpy sein) und in float32 konvertieren:
    img = image(env, obs_cache, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize)
    x = img if isinstance(img, torch.Tensor) else torch.as_tensor(img)  # → Tensor
    orig_dtype = x.dtype  # merken (z. B. torch.uint8)
    device = x.device     # CPU/GPU beibehalten
    x = x.to(torch.float32)  # in float rechnen (wichtig für Rauschen)

    # Skala: normalize=False → uint8-Interpretation (0..255), sonst 0..1
    scale = 1.0 if normalize else 255.0
    n_std = (noise_std or 0.0) * scale
    b_std = (bias_std or 0.0) * scale

    # Belichtungsfaktor (aus Licht-Intensität, auf ±15 % begrenzt)
    try:
        I = _read_light_intensity(exposure_base)
        exposure_factor = max(0.85, min(1.15, float(I) / float(exposure_base)))
    except Exception:
        exposure_factor = 1.0

    # (1) Frame-Bias: *ein* Zufallswert pro Bild (nicht pro Pixel)
    if b_std > 0:
        shape = [x.shape[0]] + [1] * (x.ndim - 1)     # (N,1,1,1)
        x = x + torch.randn(shape, device=device) * b_std

    # (2) Pixelrauschen: unabhängig per Pixel
    if n_std > 0:
        x = x + torch.randn_like(x) * n_std

    # (3a) Belichtungsabhängiger Helligkeits-Offset (additiv)
    if exposure_factor != 1.0 and exposure_brightness_gain > 0:
        x = x + (exposure_factor - 1.0) * (exposure_brightness_gain * scale)

    # (3b) Zufälliger Brightness/Contrast
    if brightness_jitter and brightness_jitter > 0:
        b = (torch.rand((x.shape[0], 1, 1, 1), device=device) * 2 - 1) * (brightness_jitter * scale)
        x = x + b
    if contrast_jitter and contrast_jitter > 0:
        c = 1.0 + (torch.rand((x.shape[0], 1, 1, 1), device=device) * 2 - 1) * contrast_jitter
        mean = x.mean(dim=(1, 2, 3), keepdim=True)  # Helligkeitsmittel pro Bild
        x = (x - mean) * c + mean

    # (4) Für Gamma/WB in [0..1] normalisieren
    y = torch.clamp(x, 0.0, 1.0) if normalize else torch.clamp(x / 255.0, 0.0, 1.0)

    # (4a) Gamma-Jitter (klein) + Belichtungs-Gamma
    if gamma_jitter and gamma_jitter > 0:
        g = 1.0 + (torch.rand((y.shape[0], 1, 1, 1), device=device) * 2 - 1) * gamma_jitter
    else:
        g = torch.ones((y.shape[0], 1, 1, 1), device=device)
    if exposure_factor != 1.0 and exposure_gamma_gain > 0:
        g = g * (1.0 + (exposure_factor - 1.0) * exposure_gamma_gain)
    y = _apply_gamma(y, g)

    # (4b) Weißabgleich (kleiner RGB-Gain, auf Mittel=1 normiert)
    if cct_jitter and cct_jitter > 0 and y.shape[-1] == 3:
        gains = torch.ones((y.shape[0], 1, 1, 3), device=device)
        deltas = (torch.rand_like(gains) * 2 - 1) * cct_jitter
        gains = gains + deltas
        gains = gains / gains.mean(dim=-1, keepdim=True).clamp(min=1e-6)  # Mittelwert normalisieren
        y = _apply_wb(y, gains)

    # zurück auf Ursprungsskala (vor Artefakten)
    x = y if normalize else y * 255.0

    # (5) selten: Dead-/Hot-Pixels
    if x.ndim == 4:  # (N,H,W,C)
        if dropout_prob and dropout_prob > 0:
            mask = (torch.rand_like(x[:, :, :, :1]) < dropout_prob)
            x = x.masked_fill(mask.expand_as(x), 0.0)
        if hotpixel_prob and hotpixel_prob > 0:
            mask = (torch.rand_like(x[:, :, :, :1]) < hotpixel_prob)
            hot_val = 0.94 * (1.0 if normalize else 255.0)  # "heiß", aber nicht ganz weiß
            x = torch.where(mask.expand_as(x), torch.full_like(x, hot_val), x)

    # (6) clampen + zurück in Ursprungstyp
    x = torch.clamp(x, 0.0, 1.0) if normalize else torch.clamp(x, 0.0, 255.0)
    if orig_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        x = x.round().to(torch.uint8)  # uint8 sauber runden
    else:
        x = x.to(orig_dtype)

    return x


def time_out(env):
    """
    Einfache Termination: Episode endet, wenn max. Länge erreicht ist.
    Für IL (Demonstrationen) reicht das oft völlig aus.
    """
    return env.episode_length_buf >= env.max_episode_length


# =============================================================================
# 2) Reset-Randomization (driftfrei) + milder Physics-Jitter
# =============================================================================

# Ablage für Basiszustände (z. B. Start-Positionen) – wird beim ersten Reset gefüllt.
_BASES: dict = {}


def _ensure_bases(env: "ManagerBasedRLEnv"):
    """
    Sammelt beim ersten Aufruf Basiswerte (hier: Positionen) der relevanten Objekte.
    Danach bleibt _BASES stabil, und wir jittern immer nur um diese Basis herum.
    """
    if _BASES.get("initialized"):
        return
    keys = list(env.scene.keys())  # echte Namen abfragen (vermeidet KeyError/Index-Verwechslungen)
    for name in ("red_apple", "orange", "bowl", "table"):
        if name in keys:
            obj = env.scene[name]
            _BASES[name] = {"pos": obj.data.root_pos_w.clone()}  
    _BASES["initialized"] = True


def _jitter_pos(base_pos: torch.Tensor, xy_range: float = 0.02, z_range: float = 0.005) -> torch.Tensor:
    """
    Kleiner Zusatz-Offset um base_pos (driftfrei).
    - xy_range, z_range sind *Maximal*-Abweichungen (Meter).
    """
    n = base_pos.shape[0]
    dx = (torch.rand((n, 1), device=base_pos.device) * 2 - 1) * xy_range
    dy = (torch.rand((n, 1), device=base_pos.device) * 2 - 1) * xy_range
    dz = (torch.rand((n, 1), device=base_pos.device) * 2 - 1) * z_range
    return base_pos + torch.cat([dx, dy, dz], dim=1)


def _set_uniform_scale(prim_path: str, scale: float):
    """
    Setzt *uniformen* USD-Scale eines Prims (rein visuell).
    Physik bleibt i. d. R. unverändert (Kollision/Masse separat).
    """
    try:
        from pxr import UsdGeom
        from omni.isaac.core.utils.prims import get_prim_at_path
        prim = get_prim_at_path(prim_path)
        if not prim:
            return
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set((scale, scale, scale))
                return
        xform.AddScaleOp().Set((scale, scale, scale))
    except Exception:
        pass


def _jitter_preview_surface(
    prim_path: str,
    base_color=(0.75, 0.75, 0.75),
    color_eps: float = 0.02,
    rough_base: float | None = None,
    rough_eps: float = 0.05,
):
    """
    Leichte Materialvariation (PreviewSurface):
      - diffuseColor: Grundfarbe ±color_eps je Kanal
      - roughness: um rough_base ±rough_eps (falls vorhanden)
    """
    try:
        from pxr import UsdShade
        from omni.isaac.core.utils.prims import get_prim_at_path

        prim = get_prim_at_path(prim_path)
        if not prim:
            return

        mat_api = UsdShade.MaterialBindingAPI(prim)
        binding = mat_api.GetDirectBinding()
        mat = binding.GetMaterial()
        if not mat:
            return

        # Shader suchen (idealerweise "PreviewSurface"; sonst ersten nehmen)
        shader = UsdShade.Shader(mat.GetPrim().GetChild("PreviewSurface"))
        if not shader:
            for child in mat.GetPrim().GetChildren():
                s = UsdShade.Shader(child)
                if s:
                    shader = s
                    break
        if not shader:
            return

        # Farbe jittern (0..1 clampen)
        r, g, b = base_color
        dr = (random.random() - 0.5) * 2 * color_eps
        dg = (random.random() - 0.5) * 2 * color_eps
        db = (random.random() - 0.5) * 2 * color_eps
        diffuse = (max(0.0, min(1.0, r + dr)),
                   max(0.0, min(1.0, g + dg)),
                   max(0.0, min(1.0, b + db)))
        if shader.GetInput("diffuseColor"):
            shader.GetInput("diffuseColor").Set(diffuse)

        # Rauheit optional jittern
        if rough_base is not None and shader.GetInput("roughness"):
            val = max(0.0, min(1.0, rough_base + (random.random() - 0.5) * 2 * rough_eps))
            shader.GetInput("roughness").Set(val)
    except Exception:
        pass


def _rotate_yaw_degrees(prim_path: str, deg: float):
    """
    Additive Z-Rotation (Yaw) um 'deg' Grad auf einem USD-Prim.
    (Wir fügen/ändern eine RotateZ-Op; addieren auf den aktuellen Wert.)
    """
    try:
        from pxr import UsdGeom
        from omni.isaac.core.utils.prims import get_prim_at_path
        prim = get_prim_at_path(prim_path)
        if not prim:
            return
        xform = UsdGeom.Xformable(prim)
        r_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                r_op = op
                break
        if r_op is None:
            r_op = xform.AddRotateZOp()
        cur = r_op.Get()
        r_op.Set((cur or 0.0) + float(deg))
    except Exception:
        pass


def _jitter_camera_extrinsics(prim_path: str, t_xy_mm: float = 1.0, t_z_mm: float = 1.0, r_deg: float = 0.3):
    """
    Minimale Kamera-Pose-Jitter:
      - Translation ±t_xy_mm (x/y) und ±t_z_mm (z) in *Millimetern*.
      - Rotation ±r_deg um x/y/z in *Grad*.
    Simuliert Montagetoleranzen/Temperaturdrift.
    """
    try:
        from pxr import UsdGeom
        from omni.isaac.core.utils.prims import get_prim_at_path
        prim = get_prim_at_path(prim_path)
        if not prim:
            return

        # Zufallsoffsets (mm → m)
        tx = (random.random() - 0.5) * 2 * (t_xy_mm / 1000.0)
        ty = (random.random() - 0.5) * 2 * (t_xy_mm / 1000.0)
        tz = (random.random() - 0.5) * 2 * (t_z_mm  / 1000.0)
        rx = (random.random() - 0.5) * 2 * r_deg
        ry = (random.random() - 0.5) * 2 * r_deg
        rz = (random.random() - 0.5) * 2 * r_deg

        xform = UsdGeom.Xformable(prim)
        t_op = None
        rx_op = ry_op = rz_op = None
        for op in xform.GetOrderedXformOps():
            t = op.GetOpType()
            if t == UsdGeom.XformOp.TypeTranslate:
                t_op = op
            elif t == UsdGeom.XformOp.TypeRotateX:
                rx_op = op
            elif t == UsdGeom.XformOp.TypeRotateY:
                ry_op = op
            elif t == UsdGeom.XformOp.TypeRotateZ:
                rz_op = op
        if t_op is None:
            t_op = xform.AddTranslateOp()
        if rx_op is None:
            rx_op = xform.AddRotateXOp()
        if ry_op is None:
            ry_op = xform.AddRotateYOp()
        if rz_op is None:
            rz_op = xform.AddRotateZOp()

        # Additiv auf bestehende Werte
        cur_t = t_op.Get() or (0.0, 0.0, 0.0)
        t_op.Set((cur_t[0] + tx, cur_t[1] + ty, cur_t[2] + tz))
        def _add(op, delta):
            cur = op.Get() or 0.0
            op.Set(float(cur) + float(delta))
        _add(rx_op, rx); _add(ry_op, ry); _add(rz_op, rz)
    except Exception:
        pass


# -------------------
# Physics-Jitter 
# -------------------

def _jitter_physx_material(prim_path: str, fric_eps: float = 0.05, rest_eps: float = 0.02):
    """
    Jittert *milde* PhysX-Material-Parameter eines USD-Prims pro Reset:
      - staticFriction  (±fric_eps, absolut)
      - dynamicFriction (±fric_eps, absolut)
      - restitution     (±rest_eps, absolut, auf [0..1] geklemmt)
    Warum mild? Große Sprünge destabilisieren Greifkontakte. Für IL reicht
    eine kleine, aber stetige Variation (Domänenlücke schließen).
    """
    try:
        from omni.isaac.core.utils.prims import get_prim_at_path
        prim = get_prim_at_path(prim_path)
        if not prim:
            return

        def _j(attr_name, base_default, eps, lo=0.0, hi=None):
            attr = prim.GetAttribute(attr_name)
            if not attr:
                return
            val = attr.Get()
            if val is None:
                val = base_default  # Fallback, wenn kein Wert gesetzt
            delta = (random.random() - 0.5) * 2 * eps
            new = val + delta
            if hi is not None:
                new = max(lo, min(hi, new))
            else:
                new = max(lo, new)
            attr.Set(float(new))

        # Häufige PhysX-Attribute (je nach Asset können Namen variieren):
        _j("physxMaterial:staticFriction",  0.6, fric_eps, lo=0.0)
        _j("physxMaterial:dynamicFriction", 0.5, fric_eps, lo=0.0)
        _j("physxMaterial:restitution",     0.1, rest_eps, lo=0.0, hi=1.0)

    except Exception:
        # Wenn das Prim kein PhysX-Material hat oder Namensschema anders ist,
        # überspringen wir still (kein harter Fehler).
        pass


# Kleine Farb-/Rauheits-Palette für den Tisch (Variation über Resets)
_TABLE_PALETTE = [
    ((0.62, 0.45, 0.28), 0.80),  # warmes Holz, rau
    ((0.45, 0.33, 0.20), 0.88),  # dunkleres Holz, rauer
    ((0.60, 0.60, 0.60), 0.70),  # neutrales Grau, glatter
]


def reset_scene_with_randomization(env: "ManagerBasedRLEnv", env_ids: torch.Tensor):
    """
    Episoden-Reset: milde, *driftfreie* Randomisierung für Realismus, ohne die
    Teleop/IL-Aufgabe zu zerstören. Alles bewusst konservativ.
    """
    _ensure_bases(env)              # Basen (Startzustände) einmalig sammeln
    keys = list(env.scene.keys())   # sichere Entity-Namensliste (vermeidet KeyErrors)

    # --- 1) Posen (driftfrei um Basis) ---
    def _jitter_entity(name: str, xy_range: float, z_range: float):
        if (name in keys) and (name in _BASES):
            obj = env.scene[name]
            base = _BASES[name]["pos"][env_ids]                 # (korr. Env-Subset)
            new_pos = _jitter_pos(base, xy_range, z_range)      # kleiner Offset
            obj.data.root_pos_w[env_ids] = new_pos              # in Buffer schreiben
            obj.write_root_state_to_sim(obj.data.root_state_w)  # in die Sim pushen

    _jitter_entity("red_apple", xy_range=0.01,  z_range=0.002)  # ±1 cm XY, ±2 mm Z
    _jitter_entity("orange",    xy_range=0.01,  z_range=0.002)
    _jitter_entity("bowl",      xy_range=0.005, z_range=0.000)  # Schüssel Z fix
    _jitter_entity("table",     xy_range=0.005, z_range=0.002)

    # --- 2) Rotationen (Yaw-only, sehr mild) ---
    try: _rotate_yaw_degrees("/World/Table", deg=(random.random() - 0.5) * 1.6)  # ±0.8°
    except Exception: pass
    try: _rotate_yaw_degrees("/World/Bowl",  deg=(random.random() - 0.5) * 4.0)  # ±2.0°
    except Exception: pass

    # --- 3) Visuelle Geometrie (Scale) ---
    try:
        _set_uniform_scale("/World/RedApple", 1.0 + (random.random() - 0.5) * 0.06)  # ±3 %
        _set_uniform_scale("/World/Orange",   1.0 + (random.random() - 0.5) * 0.08)  # ±4 %
    except Exception:
        pass

    # --- 4) Materialien (Farbe/Rauheit) ---
    try:
        _jitter_preview_surface(
            "/World/RedApple",
            base_color=(0.95, 0.10, 0.05),  # Rot
            color_eps=0.02,
            rough_base=0.70,
            rough_eps=0.05,
        )
        _jitter_preview_surface(
            "/World/Orange",
            base_color=(1.00, 0.55, 0.00),  # Orange
            color_eps=0.02,
            rough_base=0.80,
            rough_eps=0.05,
        )
        tbl_base, tbl_rough = random.choice(_TABLE_PALETTE)
        _jitter_preview_surface(
            "/World/Table",
            base_color=tbl_base,
            color_eps=0.015,
            rough_base=tbl_rough,
            rough_eps=0.04,
        )
    except Exception:
        pass

    # --- 5) Licht (Intensität/Farbe/Yaw) ---
    # Mild, damit Belichtung in image_with_noise *spürbar aber stabil* reagiert.
    try:
        from omni.isaac.core.utils.prims import get_prim_at_path
        light_prim = get_prim_at_path("/World/Light")
        if light_prim:
            base_I = 3000.0
            factor = 1.0 + (random.random() - 0.5) * 0.2   # ±10 %
            r, g, b = 0.75, 0.75, 0.75
            dr = (random.random() - 0.5) * 0.04
            dg = (random.random() - 0.5) * 0.04
            db = (random.random() - 0.5) * 0.04
            def _clamp01(x: float) -> float: return max(0.0, min(1.0, x))
            light_prim.GetAttribute("intensity").Set(base_I * factor)
            light_prim.GetAttribute("color").Set((_clamp01(r + dr), _clamp01(g + dg), _clamp01(b + db)))
            _rotate_yaw_degrees("/World/Light", deg=(random.random() - 0.5) * 10.0)  # ±5°
    except Exception:
        pass

    # --- 6) Kameras (Extrinsics) ---
    try:
        _jitter_camera_extrinsics("/World/Robot/gripper/wrist_camera", t_xy_mm=1.0, t_z_mm=1.0, r_deg=0.3)
        _jitter_camera_extrinsics("/World/Robot/base/front_camera",   t_xy_mm=1.0, t_z_mm=1.0, r_deg=0.3)
    except Exception:
        pass

    # --- 7) Physics-Jitter (*mild* halten*) ---
    try:
        _jitter_physx_material("/World/Table",    fric_eps=0.05, rest_eps=0.02)
        _jitter_physx_material("/World/RedApple", fric_eps=0.04, rest_eps=0.02)
        _jitter_physx_material("/World/Orange",   fric_eps=0.04, rest_eps=0.02)
        
    except Exception:
        pass


def reset_scene_to_default(env: "ManagerBasedRLEnv", env_ids: torch.Tensor):
    """
    Event-Hook im Reset-Modus. Wir delegieren an unsere Randomisierung.
    Vorteil: nur *eine* Stelle, die alles konsistent macht.
    """
    reset_scene_with_randomization(env, env_ids)


# =============================================================================
# 3) Optionale Terminations aus lokaler Datei 
# =============================================================================
from .terminations import * 
