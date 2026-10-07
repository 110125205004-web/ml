import shlex
from typing import Tuple

from .objects import COLOR_NAMES, KINDS, parse_color
from .scene import Scene

HELP_TEXT = [
    "commands:",
    "  add circle|square|box [size] [colour]   (short: 'circle 120 red')",
    "  select <id> | list",
    "  move <dx> <dy> | goto <x> <y>",
    "  rotate <deg> | rotate x <deg> | rotate y <deg>",
    "  scale <factor or 150%> | color <name or #rrggbb>",
    "  delete [id] | clear",
    "  save [file] | load <file>",
]


def _err(msg: str) -> Tuple[bool, str]:
    return False, msg


def run_command(text: str, scene: Scene) -> Tuple[bool, str]:
    text = (text or "").strip()
    if not text:
        return True, ""
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = text.split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("help", "?", "h"):
        return True, " | ".join(HELP_TEXT)

    if cmd in ("add", "create", "new", "make") or cmd in KINDS:
        if cmd in KINDS:
            kind, rest = cmd, args
        else:
            if not args:
                return _err("usage: add circle|square|box [size] [colour]")
            kind, rest = args[0].lower(), args[1:]
        if kind in ("cube", "rectangle", "sq", "square"):
            kind = "box" if kind in ("cube",) else ("square" if kind in ("rectangle", "sq") else kind)
        if kind not in KINDS:
            return _err(f"unknown shape '{kind}' - use {', '.join(KINDS)}")
        size, color = None, None
        for token in rest:
            if color is None and parse_color(token) is not None:
                color = parse_color(token)
            elif size is None:
                try:
                    size = float(token)
                except ValueError:
                    return _err(f"cannot read size '{token}'")
            else:
                return _err(f"unexpected argument '{token}'")
        try:
            s = scene.add(kind, size=size, color=color)
        except ValueError as e:
            return _err(str(e))
        return True, f"created #{s.id} {s.kind} (size {int(s.size)})"

    if cmd == "select":
        if not args:
            return _err("usage: select <id>|nearest")
        if args[0].lower() in ("nearest", "first"):
            if not scene.shapes:
                return _err("no objects")
            scene.ensure_active()
            a = scene.active
            if a is None:
                return _err("no objects")
            return True, f"active #{a.id}"
        try:
            sid = int(args[0].lstrip("#"))
        except ValueError:
            return _err("id must be a number")
        s = scene.select(sid)
        return (True, f"selected #{sid}") if s else _err(f"no object #{sid}")

    if cmd == "list":
        if not scene.shapes:
            return True, "no objects"
        return True, " | ".join(
            f"#{s.id} {s.kind} ({int(s.x)},{int(s.y)}) size {int(s.size)}" for s in scene.shapes
        )

    if cmd == "move":
        if len(args) != 2:
            return _err("usage: move <dx> <dy>")
        try:
            dx, dy = float(args[0]), float(args[1])
        except ValueError:
            return _err("move needs two numbers")
        if scene.active is None:
            return _err("nothing selected")
        scene.move(dx, dy)
        return True, f"moved by ({dx:g},{dy:g})"

    if cmd in ("goto", "move_to", "pos"):
        if len(args) != 2:
            return _err("usage: goto <x> <y>")
        try:
            x, y = float(args[0]), float(args[1])
        except ValueError:
            return _err("goto needs two numbers")
        if scene.active is None:
            return _err("nothing selected")
        scene.move_to(x, y)
        return True, f"moved to ({int(x)},{int(y)})"

    if cmd == "rotate":
        if len(args) == 1:
            try:
                deg = float(args[0])
            except ValueError:
                return _err("rotate needs a number")
            scene.rotate(deg)
            return True, f"rotated {deg:g} deg"
        if len(args) == 2 and args[0].lower() in ("x", "y", "z"):
            try:
                deg = float(args[1])
            except ValueError:
                return _err("rotate needs a number")
            scene.rotate(deg, axis=args[0].lower())
            return True, f"rotated {args[0].lower()} by {deg:g} deg"
        return _err("usage: rotate <deg> | rotate x|y <deg>")

    if cmd == "scale":
        if len(args) != 1:
            return _err("usage: scale <factor or 150%>")
        token = args[0]
        try:
            if token.endswith("%"):
                factor = float(token[:-1]) / 100.0
            else:
                factor = float(token)
        except ValueError:
            return _err(f"cannot read factor '{token}'")
        if factor <= 0:
            return _err("factor must be > 0")
        if scene.active is None:
            return _err("nothing selected")
        scene.scale(factor)
        return True, f"scaled x{factor:g}"

    if cmd in ("color", "colour", "fill"):
        if len(args) != 1:
            return _err(f"usage: color <{'|'.join(COLOR_NAMES)}|#rrggbb>")
        if scene.active is None:
            return _err("nothing selected")
        parsed = parse_color(args[0])
        if parsed is None:
            return _err(f"unknown colour '{args[0]}'")
        scene.set_color(parsed)
        return True, f"colour -> {args[0]}"

    if cmd in ("delete", "del", "rm", "remove"):
        sid = None
        if args:
            try:
                sid = int(args[0].lstrip("#"))
            except ValueError:
                return _err("id must be a number")
        removed = scene.delete(sid)
        return (True, f"deleted #{removed}") if removed else _err("nothing to delete")

    if cmd == "clear":
        n = scene.clear()
        return True, f"cleared {n} object(s)"

    if cmd == "save":
        path = args[0] if args else "scene.json"
        return True, "saved " + scene.save(path)

    if cmd == "load":
        if not args:
            return _err("usage: load <file>")
        try:
            other = Scene.load(args[0])
        except (OSError, ValueError, KeyError) as e:
            return _err(f"load failed: {e}")
        scene.shapes = other.shapes
        scene.width, scene.height = other.width, other.height
        scene.active_id = other.active_id
        return True, f"loaded {len(scene.shapes)} object(s)"

    if cmd in ("palette", "kind"):
        if not args or args[0].lower() not in KINDS:
            return _err(f"usage: palette {'|'.join(KINDS)}")
        return True, args[0].lower()

    return _err(f"unknown command '{cmd}' - type 'help'")
