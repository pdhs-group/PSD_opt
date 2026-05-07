from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import copy
import io
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


ROOT = Path(r"C:\Users\px2030\Code\PSD_opt\agggenerator\tests\PPT")
PPTX = ROOT / "LMC-PBE_WCPT2026_EN.pptx"
TMP = PPTX.with_suffix(".tmp.pptx")

EMU_PER_IN = 914400
SLIDE_W = 12192000
SLIDE_H = 6858000

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
}

for prefix, uri in NS.items():
    if prefix not in {"rel", "ct"}:
        ET.register_namespace(prefix, uri)
ET.register_namespace("", NS["ct"])


def qn(prefix: str, name: str) -> str:
    return f"{{{NS[prefix]}}}{name}"


def emu(inches: float) -> str:
    return str(int(round(inches * EMU_PER_IN)))


def rgb(color: str) -> str:
    return color.replace("#", "").upper()


def sub(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    el = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        if v is not None:
            el.set(k, str(v))
    return el


def max_shape_id(root: ET.Element) -> int:
    ids: list[int] = []
    for c_nv in root.findall(".//p:cNvPr", NS):
        try:
            ids.append(int(c_nv.get("id", "0")))
        except ValueError:
            pass
    return max(ids) if ids else 1


def set_existing_shape_text(root: ET.Element, shape_name_part: str, text: str) -> None:
    for shape in root.findall(".//p:sp", NS):
        c_nv = shape.find(".//p:cNvPr", NS)
        if c_nv is None:
            continue
        if shape_name_part not in c_nv.get("name", ""):
            continue
        t_nodes = shape.findall(".//a:t", NS)
        if t_nodes:
            t_nodes[0].text = text
            for t in t_nodes[1:]:
                t.text = ""
        return


def clear_content(root: ET.Element, keep_title: bool = True, keep_number: bool = True) -> ET.Element:
    sp_tree = root.find(".//p:cSld/p:spTree", NS)
    if sp_tree is None:
        raise RuntimeError("slide has no spTree")
    keep: list[ET.Element] = []
    for idx, child in enumerate(list(sp_tree)):
        if idx < 2:
            keep.append(child)
            continue
        c_nv = child.find(".//p:cNvPr", NS)
        name = c_nv.get("name", "") if c_nv is not None else ""
        if keep_number and "Slide number placeholder" in name:
            keep.append(child)
        elif keep_title and ("Titel" in name or "Title" in name):
            keep.append(child)
    for child in list(sp_tree):
        if child not in keep:
            sp_tree.remove(child)
    return sp_tree


def solid_fill(parent: ET.Element, color: str, alpha: float | None = None) -> None:
    fill = sub(parent, qn("a", "solidFill"))
    srgb = sub(fill, qn("a", "srgbClr"), val=rgb(color))
    if alpha is not None and alpha < 1:
        sub(srgb, qn("a", "alpha"), val=str(int(alpha * 100000)))


def line_style(parent: ET.Element, color: str = "#D6E2E0", width_pt: float = 1.0) -> None:
    ln = sub(parent, qn("a", "ln"), w=str(int(width_pt * 12700)))
    solid_fill(ln, color)


def shape_base(shape_id: int, name: str) -> ET.Element:
    sp = ET.Element(qn("p", "sp"))
    nv = sub(sp, qn("p", "nvSpPr"))
    sub(nv, qn("p", "cNvPr"), id=str(shape_id), name=name)
    sub(nv, qn("p", "cNvSpPr"))
    sub(nv, qn("p", "nvPr"))
    return sp


def add_rect(
    sp_tree: ET.Element,
    shape_id: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    fill_color: str = "#FFFFFF",
    line_color: str | None = "#D6E2E0",
    radius: bool = False,
    alpha: float | None = None,
) -> None:
    sp = shape_base(shape_id, name)
    sp_pr = sub(sp, qn("p", "spPr"))
    xfrm = sub(sp_pr, qn("a", "xfrm"))
    sub(xfrm, qn("a", "off"), x=emu(x), y=emu(y))
    sub(xfrm, qn("a", "ext"), cx=emu(w), cy=emu(h))
    geom = sub(sp_pr, qn("a", "prstGeom"), prst="roundRect" if radius else "rect")
    sub(geom, qn("a", "avLst"))
    solid_fill(sp_pr, fill_color, alpha)
    if line_color:
        line_style(sp_pr, line_color, 1.0)
    else:
        sub(sp_pr, qn("a", "ln")).append(ET.Element(qn("a", "noFill")))
    sp_tree.append(sp)


def add_textbox(
    sp_tree: ET.Element,
    shape_id: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str | list[str],
    font_size: int = 18,
    color: str = "#172B2A",
    bold: bool = False,
    align: str = "l",
    bullet: bool = False,
    fill_color: str | None = None,
    line_color: str | None = None,
    margin: float = 0.06,
) -> None:
    sp = shape_base(shape_id, name)
    sp_pr = sub(sp, qn("p", "spPr"))
    xfrm = sub(sp_pr, qn("a", "xfrm"))
    sub(xfrm, qn("a", "off"), x=emu(x), y=emu(y))
    sub(xfrm, qn("a", "ext"), cx=emu(w), cy=emu(h))
    geom = sub(sp_pr, qn("a", "prstGeom"), prst="rect")
    sub(geom, qn("a", "avLst"))
    if fill_color:
        solid_fill(sp_pr, fill_color)
    else:
        sub(sp_pr, qn("a", "noFill"))
    if line_color:
        line_style(sp_pr, line_color, 0.75)
    else:
        sub(sp_pr, qn("a", "ln")).append(ET.Element(qn("a", "noFill")))

    tx = sub(sp, qn("p", "txBody"))
    sub(
        tx,
        qn("a", "bodyPr"),
        wrap="square",
        lIns=emu(margin),
        rIns=emu(margin),
        tIns=emu(margin),
        bIns=emu(margin),
        anchor="t",
    )
    sub(tx, qn("a", "lstStyle"))
    lines = text if isinstance(text, list) else text.split("\n")
    for line in lines:
        p = sub(tx, qn("a", "p"))
        p_pr = sub(p, qn("a", "pPr"), algn=align)
        if bullet:
            p_pr.set("marL", "228600")
            p_pr.set("indent", "-171450")
            sub(p_pr, qn("a", "buChar"), char="•")
        r = sub(p, qn("a", "r"))
        attrs = {"lang": "en-US", "sz": str(font_size * 100)}
        if bold:
            attrs["b"] = "1"
        r_pr = sub(r, qn("a", "rPr"), **attrs)
        solid_fill(r_pr, color)
        sub(r_pr, qn("a", "latin"), typeface="Aptos")
        sub(r, qn("a", "t")).text = line
    sp_tree.append(sp)


def add_formula(
    sp_tree: ET.Element,
    shape_id: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    runs: list[tuple[str, str]],
    font_size: int = 22,
    color: str = "#172B2A",
    fill_color: str | None = None,
    line_color: str | None = None,
) -> None:
    sp = shape_base(shape_id, name)
    sp_pr = sub(sp, qn("p", "spPr"))
    xfrm = sub(sp_pr, qn("a", "xfrm"))
    sub(xfrm, qn("a", "off"), x=emu(x), y=emu(y))
    sub(xfrm, qn("a", "ext"), cx=emu(w), cy=emu(h))
    geom = sub(sp_pr, qn("a", "prstGeom"), prst="rect")
    sub(geom, qn("a", "avLst"))
    if fill_color:
        solid_fill(sp_pr, fill_color)
    else:
        sub(sp_pr, qn("a", "noFill"))
    if line_color:
        line_style(sp_pr, line_color, 0.75)
    else:
        sub(sp_pr, qn("a", "ln")).append(ET.Element(qn("a", "noFill")))

    tx = sub(sp, qn("p", "txBody"))
    sub(tx, qn("a", "bodyPr"), wrap="none", lIns="0", rIns="0", tIns="0", bIns="0", anchor="ctr")
    sub(tx, qn("a", "lstStyle"))
    p = sub(tx, qn("a", "p"))
    sub(p, qn("a", "pPr"), algn="ctr")
    for text_value, role in runs:
        r = sub(p, qn("a", "r"))
        attrs = {"lang": "en-US", "sz": str(font_size * 100)}
        if role == "sub":
            attrs["baseline"] = "-25000"
            attrs["sz"] = str(int(font_size * 0.82 * 100))
        elif role == "sup":
            attrs["baseline"] = "30000"
            attrs["sz"] = str(int(font_size * 0.82 * 100))
        r_pr = sub(r, qn("a", "rPr"), **attrs)
        solid_fill(r_pr, color)
        sub(r_pr, qn("a", "latin"), typeface="Cambria Math")
        sub(r, qn("a", "t")).text = text_value
    sp_tree.append(sp)


def add_arrow(
    sp_tree: ET.Element,
    shape_id: int,
    name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#009682",
    width_pt: float = 2.0,
) -> None:
    cxn = ET.Element(qn("p", "cxnSp"))
    nv = sub(cxn, qn("p", "nvCxnSpPr"))
    sub(nv, qn("p", "cNvPr"), id=str(shape_id), name=name)
    sub(nv, qn("p", "cNvCxnSpPr"))
    sub(nv, qn("p", "nvPr"))
    sp_pr = sub(cxn, qn("p", "spPr"))
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    xfrm = sub(sp_pr, qn("a", "xfrm"))
    if x2 < x1:
        xfrm.set("flipH", "1")
    if y2 < y1:
        xfrm.set("flipV", "1")
    sub(xfrm, qn("a", "off"), x=emu(x), y=emu(y))
    sub(xfrm, qn("a", "ext"), cx=emu(max(w, 0.01)), cy=emu(max(h, 0.01)))
    geom = sub(sp_pr, qn("a", "prstGeom"), prst="line")
    sub(geom, qn("a", "avLst"))
    ln = sub(sp_pr, qn("a", "ln"), w=str(int(width_pt * 12700)))
    solid_fill(ln, color)
    sub(ln, qn("a", "tailEnd"), type="triangle")
    sp_tree.append(cxn)


@dataclass
class SlideCtx:
    slide_no: int
    root: ET.Element
    rels: ET.Element
    sp_tree: ET.Element
    next_id: int
    media_payloads: dict[str, bytes] = field(default_factory=dict)

    def nid(self) -> int:
        self.next_id += 1
        return self.next_id

    def add_rel(self, target_name: str) -> str:
        existing = {
            rel.get("Id")
            for rel in self.rels.findall("rel:Relationship", NS)
            if rel.get("Id", "").startswith("rId")
        }
        max_rid = 0
        for rid in existing:
            try:
                max_rid = max(max_rid, int(rid[3:]))
            except Exception:
                pass
        rid = f"rId{max_rid + 1}"
        rel = sub(
            self.rels,
            qn("rel", "Relationship"),
            Id=rid,
            Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            Target=f"../media/{target_name}",
        )
        return rid


def remove_codex_rels(rels: ET.Element) -> None:
    for rel in list(rels):
        target = rel.get("Target", "")
        if "../media/codex_" in target:
            rels.remove(rel)


def add_picture(
    ctx: SlideCtx,
    name: str,
    asset: Path,
    x: float,
    y: float,
    box_w: float,
    box_h: float,
    contain: bool = True,
    bg: bool = True,
) -> None:
    suffix = asset.suffix.lower()
    target = f"codex_{asset.stem}{suffix}"
    data = asset.read_bytes()
    ctx.media_payloads[target] = data
    with Image.open(io.BytesIO(data)) as im:
        iw, ih = im.size
    if contain:
        scale = min(box_w / iw, box_h / ih)
        w = iw * scale
        h = ih * scale
        x = x + (box_w - w) / 2
        y = y + (box_h - h) / 2
    else:
        w, h = box_w, box_h
    if bg:
        add_rect(ctx.sp_tree, ctx.nid(), f"{name} background", x - 0.06, y - 0.06, w + 0.12, h + 0.12, "#FFFFFF", "#D8E7E4", True)
    rid = ctx.add_rel(target)
    pic = ET.Element(qn("p", "pic"))
    nv = sub(pic, qn("p", "nvPicPr"))
    c_nv = sub(nv, qn("p", "cNvPr"), id=str(ctx.nid()), name=name, descr=asset.name)
    c_nv_pic = sub(nv, qn("p", "cNvPicPr"))
    sub(c_nv_pic, qn("a", "picLocks"), noChangeAspect="1")
    sub(nv, qn("p", "nvPr"))
    blip_fill = sub(pic, qn("p", "blipFill"))
    sub(blip_fill, qn("a", "blip"), **{qn("r", "embed"): rid})
    stretch = sub(blip_fill, qn("a", "stretch"))
    sub(stretch, qn("a", "fillRect"))
    sp_pr = sub(pic, qn("p", "spPr"))
    xfrm = sub(sp_pr, qn("a", "xfrm"))
    sub(xfrm, qn("a", "off"), x=emu(x), y=emu(y))
    sub(xfrm, qn("a", "ext"), cx=emu(w), cy=emu(h))
    geom = sub(sp_pr, qn("a", "prstGeom"), prst="rect")
    sub(geom, qn("a", "avLst"))
    ctx.sp_tree.append(pic)


def section_label(ctx: SlideCtx, label: str) -> None:
    add_textbox(ctx.sp_tree, ctx.nid(), "section label", 0.48, 0.86, 2.2, 0.25, label.upper(), 9, "#009682", True, margin=0)


def slide1(root: ET.Element) -> None:
    set_existing_shape_text(root, "Textplatzhalter", "Microstructure-aware breakage kernels for Monte Carlo PBE")
    # Keep the conference line and author line, but make the subtitle more explicit.
    for shape in root.findall(".//p:sp", NS):
        c_nv = shape.find(".//p:cNvPr", NS)
        if c_nv is not None and "Inhaltsplatzhalter" in c_nv.get("name", ""):
            t_nodes = shape.findall(".//a:t", NS)
            if t_nodes:
                t_nodes[0].text = "Aggregate generator + LMC fragmentation + energy-rate closure"
                if len(t_nodes) > 1:
                    t_nodes[1].text = "Haoran Ji, Simon Buchheiser | WCPT 2026"
                for t in t_nodes[2:]:
                    t.text = ""


def slide2(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "10-minute arc")
    clear_content(ctx.root)
    add_textbox(ctx.sp_tree, ctx.nid(), "agenda intro", 0.75, 1.15, 6.4, 0.6, "One question: how can a PBE see particle microstructure?", 28, "#172B2A", True, margin=0)
    items = [
        "1  Motivation: breakage kernels miss internal structure",
        "2  Aggregate generator: morphology + material field",
        "3  LMC: crack growth, fragments, fracture energy",
        "4  Energy-rate closure for Monte Carlo PBE",
        "5  Validation signals and next steps",
    ]
    add_textbox(ctx.sp_tree, ctx.nid(), "agenda items", 1.05, 2.15, 6.6, 3.7, items, 22, "#172B2A", False, bullet=False, margin=0)
    add_rect(ctx.sp_tree, ctx.nid(), "agenda visual rail", 8.0, 1.35, 3.9, 4.4, "#EEF7F5", "#B8DCD6", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "agenda formula", 8.35, 1.82, 3.2, 0.75, "structure z", 30, "#009682", True, "ctr", margin=0)
    add_arrow(ctx.sp_tree, ctx.nid(), "agenda arrow 1", 9.95, 2.75, 9.95, 3.35, "#009682", 2.2)
    add_textbox(ctx.sp_tree, ctx.nid(), "agenda formula2", 8.38, 3.45, 3.15, 0.75, "LMC samples", 30, "#009682", True, "ctr", margin=0)
    add_arrow(ctx.sp_tree, ctx.nid(), "agenda arrow 2", 9.95, 4.4, 9.95, 5.0, "#009682", 2.2)
    add_textbox(ctx.sp_tree, ctx.nid(), "agenda formula3", 8.2, 5.05, 3.55, 0.65, "PBE kernel", 30, "#009682", True, "ctr", margin=0)


def slide3(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "Motivation: microstructure disappears inside empirical kernels")
    clear_content(ctx.root)
    section_label(ctx, "motivation")
    add_textbox(ctx.sp_tree, ctx.nid(), "core claim", 0.65, 1.08, 9.5, 0.58, "Breakage rate and daughter fragments depend on the internal graph, not only particle size.", 26, "#172B2A", True, margin=0)
    cols = [
        ("Classical kernels", "low cost\nempirical coefficients\nweak structural resolution", "#F4B63F"),
        ("DEM", "high fidelity\nexplicit contact physics\nhard to call inside PBE", "#6FA8DC"),
        ("This framework", "sampled microstructure\nLMC fracture statistics\nPBE-ready closure", "#009682"),
    ]
    x0 = 0.7
    for i, (head, body, col) in enumerate(cols):
        x = x0 + i * 4.0
        add_rect(ctx.sp_tree, ctx.nid(), f"motivation block {i}", x, 2.35, 3.45, 2.45, "#FFFFFF", col, True)
        add_textbox(ctx.sp_tree, ctx.nid(), f"motivation head {i}", x + 0.22, 2.62, 3.0, 0.35, head, 20, col, True, margin=0)
        add_textbox(ctx.sp_tree, ctx.nid(), f"motivation body {i}", x + 0.28, 3.25, 2.75, 1.1, body.split("\n"), 16, "#172B2A", False, bullet=True, margin=0)
        if i < 2:
            add_arrow(ctx.sp_tree, ctx.nid(), f"motivation arrow {i}", x + 3.58, 3.55, x + 3.9, 3.55, "#8AA29E", 2.0)
    add_textbox(ctx.sp_tree, ctx.nid(), "bottom takeaway", 1.0, 5.6, 10.8, 0.5, "Design target: keep PBE fast while letting aggregate morphology and material distribution change the breakage statistics.", 19, "#3F5551", False, "ctr", margin=0)


def slide4(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "Framework: external microstructure module for PBE kernels")
    clear_content(ctx.root)
    section_label(ctx, "model overview")
    assets = [
        (ROOT / "mptsa2d_51.png", "Aggregate state z", "Df, MAS, FRAC_A, NP"),
        (ROOT / "udp_result.png", "LMC fracture samples", "fragments + Ereq"),
        (ROOT / "fragments_distribution.png", "PBE kernel input", "b(vA,vB|z), a(V,z)"),
    ]
    xs = [0.65, 4.75, 8.85]
    for i, (asset, head, subline) in enumerate(assets):
        add_picture(ctx, f"framework image {i}", asset, xs[i], 1.35, 3.35, 3.15)
        add_textbox(ctx.sp_tree, ctx.nid(), f"framework head {i}", xs[i], 4.72, 3.35, 0.35, head, 20, "#009682", True, "ctr", margin=0)
        add_textbox(ctx.sp_tree, ctx.nid(), f"framework sub {i}", xs[i], 5.16, 3.35, 0.35, subline, 15, "#536864", False, "ctr", margin=0)
        if i < 2:
            add_arrow(ctx.sp_tree, ctx.nid(), f"framework arrow {i}", xs[i] + 3.48, 2.88, xs[i] + 3.95, 2.88, "#009682", 2.5)
    add_textbox(ctx.sp_tree, ctx.nid(), "framework bottom", 1.1, 6.25, 11.0, 0.42, "The PBE never resolves cracks directly; it queries a statistically sampled microstructure module.", 19, "#172B2A", True, "ctr", margin=0)


def slide5(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "Aggregate generator: morphology first, material field second")
    clear_content(ctx.root)
    section_label(ctx, "aggregate generator")
    add_picture(ctx, "mptsa animation", ROOT / "mptsa2d.gif", 0.7, 1.25, 4.55, 4.55)
    add_picture(ctx, "material mix animation", ROOT / "material_mix.gif", 5.35, 1.25, 4.55, 4.55)
    add_textbox(ctx.sp_tree, ctx.nid(), "mptsa label", 0.85, 5.92, 4.2, 0.36, "MPTSA growth: target N and Df", 18, "#009682", True, "ctr", margin=0)
    add_textbox(ctx.sp_tree, ctx.nid(), "mix label", 5.5, 5.92, 4.2, 0.36, "Material assignment: exact fraction + MAS", 18, "#009682", True, "ctr", margin=0)
    add_arrow(ctx.sp_tree, ctx.nid(), "generator arrow", 4.95, 3.55, 5.32, 3.55, "#009682", 2.6)
    add_rect(ctx.sp_tree, ctx.nid(), "generator parameter rail", 10.2, 1.3, 2.45, 4.45, "#EEF7F5", "#B8DCD6", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "generator parameter title", 10.45, 1.66, 1.95, 0.38, "control knobs", 17, "#009682", True, "ctr", margin=0)
    add_textbox(ctx.sp_tree, ctx.nid(), "generator parameters", 10.48, 2.28, 1.85, 1.75, ["Df: morphology", "MAS: mixing", "FRAC_A: composition", "NP: grid size"], 15, "#172B2A", False, bullet=True, margin=0)
    add_textbox(ctx.sp_tree, ctx.nid(), "generator output", 10.45, 4.7, 1.95, 0.65, "output:\nlabeled lattice", 18, "#172B2A", True, "ctr", margin=0)


def slide6(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "LMC: crack propagation on material-dependent bonds")
    clear_content(ctx.root)
    section_label(ctx, "lmc fragmentation")
    add_picture(ctx, "udp crack animation", ROOT / "udp_crack.gif", 0.75, 1.05, 5.75, 5.75)
    add_picture(ctx, "udp final frame", ROOT / "udp_crack_g0_f0_c3_s13_88.png", 6.9, 1.08, 2.45, 2.45)
    add_textbox(ctx.sp_tree, ctx.nid(), "lmc final label", 6.9, 3.62, 2.45, 0.35, "accepted crack path", 16, "#009682", True, "ctr", margin=0)
    add_rect(ctx.sp_tree, ctx.nid(), "lmc workflow rail", 9.72, 1.1, 2.8, 4.95, "#FFFFFF", "#B8DCD6", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "lmc workflow title", 9.95, 1.45, 2.35, 0.35, "event loop", 19, "#009682", True, "ctr", margin=0)
    steps = ["seed crack", "sample next bond", "update graph", "stop at target fragments"]
    y = 2.05
    for i, step in enumerate(steps):
        add_rect(ctx.sp_tree, ctx.nid(), f"lmc step {i}", 10.05, y + i * 0.82, 2.15, 0.44, "#EEF7F5", "#CFE5E0", True)
        add_textbox(ctx.sp_tree, ctx.nid(), f"lmc step text {i}", 10.15, y + 0.08 + i * 0.82, 1.95, 0.22, step, 14, "#172B2A", True, "ctr", margin=0)
        if i < len(steps) - 1:
            add_arrow(ctx.sp_tree, ctx.nid(), f"lmc step arrow {i}", 11.12, y + 0.48 + i * 0.82, 11.12, y + 0.77 + i * 0.82, "#009682", 1.4)
    add_textbox(ctx.sp_tree, ctx.nid(), "lmc output note", 6.82, 5.0, 2.7, 0.74, "bond strengths convert crack geometry into required fracture energy", 18, "#172B2A", True, "ctr", margin=0)


def slide7(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "LMC outputs: daughter distribution and required fracture energy")
    clear_content(ctx.root)
    section_label(ctx, "kernel data")
    add_picture(ctx, "fragment result", ROOT / "udp_result.png", 0.75, 1.12, 3.9, 4.3)
    add_picture(ctx, "fragment distribution plot", ROOT / "fragments_distribution.png", 4.9, 1.12, 3.9, 4.3)
    add_rect(ctx.sp_tree, ctx.nid(), "equation panel", 9.05, 1.25, 3.25, 3.95, "#EEF7F5", "#B8DCD6", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "eq title", 9.38, 1.58, 2.55, 0.32, "sampled quantities", 18, "#009682", True, "ctr", margin=0)
    add_formula(ctx.sp_tree, ctx.nid(), "fragment formula", 9.35, 2.17, 2.55, 0.55, [("b(v", "normal"), ("A", "sub"), (", v", "normal"), ("B", "sub"), (" | z)", "normal")], 24)
    add_textbox(ctx.sp_tree, ctx.nid(), "fragment eq note", 9.4, 2.82, 2.45, 0.38, "daughter-state distribution", 13, "#536864", False, "ctr", margin=0)
    add_formula(ctx.sp_tree, ctx.nid(), "energy formula", 9.0, 3.56, 3.28, 0.6, [("E", "normal"), ("req", "sub"), ("(z) = Σ", "normal"), ("b∈C(z)", "sub"), (" G", "normal"), ("b", "sub")], 20)
    add_textbox(ctx.sp_tree, ctx.nid(), "energy eq note", 9.32, 4.18, 2.65, 0.42, "broken-bond energy demand", 13, "#536864", False, "ctr", margin=0)
    add_textbox(ctx.sp_tree, ctx.nid(), "lmc output bottom", 1.0, 6.12, 11.0, 0.42, "Repeated LMC runs turn one lattice into a stochastic breakage kernel sample.", 19, "#172B2A", True, "ctr", margin=0)


def slide8(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "Breakage-rate model: energy demand vs available input")
    clear_content(ctx.root)
    section_label(ctx, "rate closure")
    add_textbox(ctx.sp_tree, ctx.nid(), "rate claim", 0.65, 1.05, 6.7, 0.62, "The rate model maps an LMC energy requirement to a PBE event intensity.", 27, "#172B2A", True, margin=0)
    add_rect(ctx.sp_tree, ctx.nid(), "formula block 1", 0.8, 2.05, 3.1, 0.82, "#FFFFFF", "#B8DCD6", True)
    add_formula(ctx.sp_tree, ctx.nid(), "formula req", 1.0, 2.18, 2.7, 0.46, [("E", "normal"), ("req", "sub"), (" = Σ", "normal"), ("b∈C", "sub"), (" G", "normal"), ("b", "sub")], 19)
    add_rect(ctx.sp_tree, ctx.nid(), "formula block 2", 0.8, 3.25, 3.1, 0.82, "#FFFFFF", "#B8DCD6", True)
    add_formula(ctx.sp_tree, ctx.nid(), "formula ein", 1.05, 3.38, 2.6, 0.46, [("E", "normal"), ("in", "sub"), ("(V) = λV", "normal"), ("α", "sup")], 20)
    add_rect(ctx.sp_tree, ctx.nid(), "formula block 3", 4.7, 2.65, 2.2, 0.92, "#009682", "#009682", True)
    add_formula(ctx.sp_tree, ctx.nid(), "formula rho", 4.82, 2.8, 1.95, 0.52, [("ρ = E", "normal"), ("in", "sub"), (" / E", "normal"), ("req", "sub")], 18, "#FFFFFF")
    add_arrow(ctx.sp_tree, ctx.nid(), "formula arrow 1", 3.95, 2.46, 4.65, 2.93, "#009682", 2.0)
    add_arrow(ctx.sp_tree, ctx.nid(), "formula arrow 2", 3.95, 3.66, 4.65, 3.2, "#009682", 2.0)
    add_rect(ctx.sp_tree, ctx.nid(), "formula block 4", 7.35, 2.25, 4.7, 1.55, "#EEF7F5", "#B8DCD6", True)
    add_formula(ctx.sp_tree, ctx.nid(), "formula rate", 7.58, 2.45, 4.28, 0.55, [("a(V,z) = k", "normal"), ("0", "sub"), ("V", "normal"), ("γ", "sup"), (" Φ(logρ; θ)", "normal")], 20)
    add_textbox(ctx.sp_tree, ctx.nid(), "formula 4 note", 7.75, 3.14, 3.95, 0.3, "calibrated monotone transfer, not a fixed physics law", 13, "#536864", False, "ctr", margin=0)
    add_arrow(ctx.sp_tree, ctx.nid(), "formula arrow 3", 6.95, 3.1, 7.28, 3.1, "#009682", 2.0)
    add_picture(ctx, "energy distribution", ROOT / "Energy_Distribution.png", 6.95, 4.1, 4.55, 2.0)
    add_textbox(ctx.sp_tree, ctx.nid(), "rate bottom", 0.95, 6.22, 11.3, 0.42, "Practical implementation: LMC provides E_req samples; an MLP surrogate can replace repeated energy scans inside MCPBE.", 18, "#172B2A", True, "ctr", margin=0)


def slide9(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "Validation setup: ultrasonic fragmentation as a first closure test")
    clear_content(ctx.root)
    section_label(ctx, "validation")
    curve_img = ROOT / "extracted_slide9_media" / "rId9_image23.png"
    add_rect(ctx.sp_tree, ctx.nid(), "experiment schematic rail", 0.65, 1.15, 5.15, 2.35, "#FFFFFF", "#B8DCD6", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "exp node 1", 0.95, 1.82, 1.35, 0.45, "energy input", 15, "#009682", True, "ctr", fill_color="#EEF7F5", line_color="#CFE5E0")
    add_textbox(ctx.sp_tree, ctx.nid(), "exp node 2", 2.6, 1.82, 1.35, 0.45, "aggregate breakage", 15, "#009682", True, "ctr", fill_color="#EEF7F5", line_color="#CFE5E0")
    add_textbox(ctx.sp_tree, ctx.nid(), "exp node 3", 4.25, 1.82, 1.05, 0.45, "ADC PSD", 15, "#009682", True, "ctr", fill_color="#EEF7F5", line_color="#CFE5E0")
    add_arrow(ctx.sp_tree, ctx.nid(), "exp arrow 1", 2.32, 2.05, 2.58, 2.05, "#009682", 1.8)
    add_arrow(ctx.sp_tree, ctx.nid(), "exp arrow 2", 3.98, 2.05, 4.23, 2.05, "#009682", 1.8)
    add_textbox(ctx.sp_tree, ctx.nid(), "exp note", 1.0, 2.75, 4.3, 0.32, "closed batch system | carbon black | two input powers", 14, "#536864", False, "ctr", margin=0)
    add_picture(ctx, "validation curve", curve_img, 7.95, 1.12, 4.35, 2.85)
    add_rect(ctx.sp_tree, ctx.nid(), "validation setup", 0.85, 4.1, 3.2, 1.55, "#EEF7F5", "#B8DCD6", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "validation setup title", 1.08, 4.36, 2.7, 0.28, "experimental input", 16, "#009682", True, "ctr", margin=0)
    add_textbox(ctx.sp_tree, ctx.nid(), "validation setup body", 1.12, 4.78, 2.62, 0.58, ["ultrasonic fragmentation", "carbon black", "ADC PSD measurement"], 13, "#172B2A", False, bullet=True, margin=0)
    add_rect(ctx.sp_tree, ctx.nid(), "validation calibration", 4.75, 4.1, 3.15, 1.55, "#FFFFFF", "#B8DCD6", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "validation calibration title", 4.98, 4.36, 2.7, 0.28, "calibration target", 16, "#009682", True, "ctr", margin=0)
    add_textbox(ctx.sp_tree, ctx.nid(), "validation calibration body", 5.05, 4.78, 2.55, 0.58, ["two input powers", "λ1 ≈ 2.16e-7", "λ2 ≈ 2.11e-7"], 13, "#172B2A", False, bullet=True, margin=0)
    add_rect(ctx.sp_tree, ctx.nid(), "validation result", 8.65, 4.1, 3.15, 1.55, "#FFFFFF", "#B8DCD6", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "validation result title", 8.88, 4.36, 2.7, 0.28, "readout", 16, "#009682", True, "ctr", margin=0)
    add_textbox(ctx.sp_tree, ctx.nid(), "validation result body", 8.95, 4.78, 2.55, 0.58, ["PSD trend reproduced", "energy-rate closure consistent", "residual size-limit bias"], 13, "#172B2A", False, bullet=True, margin=0)


def slide10(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "Validation signals: energy and fragment statistics")
    clear_content(ctx.root)
    section_label(ctx, "model behavior")
    add_picture(ctx, "energy distribution large", ROOT / "Energy_Distribution.png", 0.75, 1.25, 5.3, 3.75)
    add_picture(ctx, "fragment distribution large", ROOT / "fragments_distribution.png", 6.45, 1.0, 4.25, 4.25)
    add_textbox(ctx.sp_tree, ctx.nid(), "signal 1", 1.0, 5.45, 4.75, 0.42, "Energy demand: broad right-skewed distribution from microstructure-dependent fracture paths", 17, "#172B2A", True, "ctr", margin=0)
    add_textbox(ctx.sp_tree, ctx.nid(), "signal 2", 6.45, 5.45, 4.25, 0.42, "Fragment kernel: joint daughter-volume statistics, not a single deterministic split", 17, "#172B2A", True, "ctr", margin=0)
    add_rect(ctx.sp_tree, ctx.nid(), "model behavior note", 10.95, 1.35, 1.55, 4.15, "#EEF7F5", "#B8DCD6", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "model behavior note text", 11.12, 1.9, 1.22, 2.7, "Reusable kernel data for MCPBE sampling", 18, "#009682", True, "ctr", margin=0)


def slide11(ctx: SlideCtx) -> None:
    set_existing_shape_text(ctx.root, "Titel", "Next steps: validation for structured aggregates")
    clear_content(ctx.root)
    section_label(ctx, "next steps")
    add_textbox(ctx.sp_tree, ctx.nid(), "next claim", 0.75, 1.08, 7.6, 0.55, "The model path is ready; the limiting step is multidimensional experimental evidence.", 27, "#172B2A", True, margin=0)
    lanes = [
        ("Structure scan", ["Df / MAS / FRAC_A", "aggregate-pool coverage", "sensitivity maps"]),
        ("Physics closure", ["LMC parameter validation", "energy-transfer function", "weighted MCPBE"]),
        ("Measurements", ["2D / multimaterial PSD", "3D-printed aggregates", "DEM comparison"]),
    ]
    for i, (head, items) in enumerate(lanes):
        x = 0.85 + i * 4.0
        add_rect(ctx.sp_tree, ctx.nid(), f"next lane {i}", x, 2.1, 3.25, 2.7, "#FFFFFF", "#B8DCD6", True)
        add_textbox(ctx.sp_tree, ctx.nid(), f"next lane head {i}", x + 0.2, 2.42, 2.85, 0.3, head, 18, "#009682", True, "ctr", margin=0)
        add_textbox(ctx.sp_tree, ctx.nid(), f"next lane items {i}", x + 0.35, 3.0, 2.55, 0.95, items, 14, "#172B2A", False, bullet=True, margin=0)
    add_rect(ctx.sp_tree, ctx.nid(), "open challenge", 2.1, 5.45, 9.1, 0.72, "#FFF3C4", "#F4B63F", True)
    add_textbox(ctx.sp_tree, ctx.nid(), "open challenge text", 2.35, 5.66, 8.6, 0.24, "Open challenge: composition-size distributions for multi-material aggregates", 17, "#6A4A00", True, "ctr", margin=0)


def update_content_types(root: ET.Element) -> None:
    defaults = {d.get("Extension", "").lower() for d in root.findall("ct:Default", NS)}
    if "gif" not in defaults:
        sub(root, qn("ct", "Default"), Extension="gif", ContentType="image/gif")
    if "png" not in defaults:
        sub(root, qn("ct", "Default"), Extension="png", ContentType="image/png")


def build() -> None:
    slide_roots: dict[int, ET.Element] = {}
    slide_rels: dict[int, ET.Element] = {}
    content_types: ET.Element | None = None
    media_payloads: dict[str, bytes] = {}

    with ZipFile(PPTX, "r") as zin:
        for no in range(1, 14):
            slide_roots[no] = ET.fromstring(zin.read(f"ppt/slides/slide{no}.xml"))
            rel_name = f"ppt/slides/_rels/slide{no}.xml.rels"
            if rel_name in zin.namelist():
                slide_rels[no] = ET.fromstring(zin.read(rel_name))
            else:
                slide_rels[no] = ET.Element(qn("rel", "Relationships"))
        content_types = ET.fromstring(zin.read("[Content_Types].xml"))

    slide1(slide_roots[1])

    builders = {
        2: slide2,
        3: slide3,
        4: slide4,
        5: slide5,
        6: slide6,
        7: slide7,
        8: slide8,
        9: slide9,
        10: slide10,
        11: slide11,
    }
    for no, builder in builders.items():
        remove_codex_rels(slide_rels[no])
        sp_tree = clear_content(slide_roots[no])
        ctx = SlideCtx(no, slide_roots[no], slide_rels[no], sp_tree, max_shape_id(slide_roots[no]))
        builder(ctx)
        media_payloads.update(ctx.media_payloads)

    if content_types is None:
        raise RuntimeError("missing content types")
    update_content_types(content_types)

    with ZipFile(PPTX, "r") as zin, ZipFile(TMP, "w", ZIP_DEFLATED) as zout:
        replaced = {"[Content_Types].xml"}
        for no in range(1, 14):
            replaced.add(f"ppt/slides/slide{no}.xml")
            replaced.add(f"ppt/slides/_rels/slide{no}.xml.rels")
        skip_media = {f"ppt/media/{name}" for name in media_payloads}
        for item in zin.infolist():
            if item.filename in replaced or item.filename in skip_media:
                continue
            zout.writestr(item, zin.read(item.filename))
        zout.writestr("[Content_Types].xml", ET.tostring(content_types, encoding="utf-8", xml_declaration=True))
        for no in range(1, 14):
            zout.writestr(f"ppt/slides/slide{no}.xml", ET.tostring(slide_roots[no], encoding="utf-8", xml_declaration=True))
            zout.writestr(f"ppt/slides/_rels/slide{no}.xml.rels", ET.tostring(slide_rels[no], encoding="utf-8", xml_declaration=True))
        for name, data in media_payloads.items():
            zout.writestr(f"ppt/media/{name}", data)

    with ZipFile(TMP, "r") as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"bad zip member: {bad}")
        for name in z.namelist():
            if name.endswith(".xml"):
                ET.fromstring(z.read(name))
    shutil.move(str(TMP), str(PPTX))
    print(f"updated {PPTX}")
    print(f"embedded media: {len(media_payloads)}")


if __name__ == "__main__":
    build()
