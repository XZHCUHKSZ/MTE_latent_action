"""Paper-facing predictive-comparison names; frozen implementation IDs stay stable.

PC is the predictive-comparison framework. MTE names its matched difference,
not another method. Resolution changes naming only, never training settings.
"""
import re

FAMILY = {
    "pc_mean": "edge_cara",
    "pc_set": "edge_cara_mobius_simple",
    "pc_graph": "edge_cara_mobius_graph",
    "pc_sweep": "edge_cara_mobius_tree",
    "pc_field": "edge_cara_mif",
}
LABELS = {key: "PC-" + key[3:].title() for key in FAMILY}

def resolve_method(name, interface="method"):
    """Resolve a public name before dispatch, preserving all legacy IDs.

    interface=method accepts bare encoders (plus explicit Mean horizons).
    visual/coupled also accept -Solo/-Aux and return original composition IDs.
    mpe accepts the corrected numeric protocol IDs; Mean requires h2/h3.
    Horizons are never silently dropped or reassigned.
    """
    if interface not in {"method", "visual", "coupled", "mpe"}:
        raise ValueError("Unknown method interface: " + interface)
    if not isinstance(name, str):
        raise TypeError("Method name must be a string")
    if not name.lower().startswith(("pc-", "pc_")):
        return name
    if interface == "mpe":
        key = name.lower().replace("-", "_")
        route = "aux" if key.endswith("_aux") else "solo" if key.endswith("_solo") else None
        base = key.rsplit("_", 1)[0] if route else key
        native = resolve_control(base)
        if native.startswith("edge_h1") or native == "edge":
            raise ValueError("MPE Mean requires its explicit h2 or h3 protocol")
        if not re.fullmatch(r"(?:edge_h[23](?:_root)?|(?:simple|graph|tree|mif)(?:_(?:raw|complement))?)", native):
            raise ValueError("Not a registered MPE arm: " + name)
        return ("base+" if route == "aux" else "") + native
    key = name.lower().replace("-", "_")
    match = re.fullmatch(r"(pc_mean|pc_set|pc_graph|pc_sweep|pc_field)(_h[123])?(_solo|_aux)?", key)
    if not match:
        raise ValueError("Unknown predictive-comparison name: " + name)
    family, horizon, composition = match.groups()
    if horizon and (family != "pc_mean" or interface != "method"):
        raise ValueError("Explicit horizons are numeric Mean protocol IDs only")
    native = FAMILY[family] + (horizon or "")
    if composition:
        if interface == "method":
            raise ValueError("Solo/Aux belongs to a composition interface")
        prefixes = {"visual": {"_solo": "anchor_solo_", "_aux": "anchor_plus_"},
                    "coupled": {"_solo": "solo_", "_aux": "aux_"}}
        return prefixes[interface][composition] + native
    return native

def paper_name(name, interface="method"):
    """Display a legacy encoder/composition ID without changing stored evidence."""
    if interface == "mpe":
        auxiliary = name.startswith("base+")
        short = name[5:] if auxiliary else name
        mapping = {"simple": "PC-Set", "graph": "PC-Graph", "tree": "PC-Sweep", "mif": "PC-Field",
                   "edge_h2": "PC-Mean-h2", "edge_h3": "PC-Mean-h3"}
        return mapping[short] + ("-Aux" if auxiliary else "-Solo") if short in mapping else name
    if interface not in {"method", "control"}:
        raise ValueError("Unknown display interface: " + interface)
    original = name
    route = ""
    for prefix, suffix in [("anchor_solo_", "-Solo"), ("anchor_plus_", "-Aux"),
                           ("solo_", "-Solo"), ("aux_", "-Aux")]:
        if name.startswith(prefix):
            name, route = name[len(prefix):], suffix
            break
    for public, native in sorted(FAMILY.items(), key=lambda item: -len(item[1])):
        if name == native:
            return LABELS[public] + route
        if public == "pc_mean" and name in {native + "_h1", native + "_h2", native + "_h3"}:
            return LABELS[public] + "-" + name[-2:] + route
    return original

SHORT = {"pc_mean": "edge", "pc_set": "simple", "pc_graph": "graph",
         "pc_sweep": "tree", "pc_field": "mif"}
# Control suffixes are labels for existing settings, never new hyperparameters.
CONTROL_SUFFIX = re.compile(
    r"(?:h[123](?:_root)?|matched|raw|complement|edge|pair|formal|"
    r"(?:solo|aux)(?:_(?:pretrained|random)_(?:frozen|trainable|baseonly|auxonly))?)?")


def resolve_control(name):
    """Resolve short family/structural/branch IDs, preserving every qualifier."""
    if not name.lower().startswith(("pc-", "pc_")):
        return name
    key = name.lower().replace("-", "_")
    match = re.fullmatch(r"(pc_mean|pc_set|pc_graph|pc_sweep|pc_field)(?:_(.*))?", key)
    if not match:
        raise ValueError("Unknown PC control: " + name)
    family, suffix = match.groups()
    suffix = suffix or ""
    if not CONTROL_SUFFIX.fullmatch(suffix):
        raise ValueError("Unknown PC control qualifier: " + name)
    if family == "pc_mean" and not re.fullmatch(r"h[123](?:_root)?", suffix):
        raise ValueError("Short Mean control requires its explicit horizon")
    if family != "pc_mean" and suffix.startswith("h"):
        raise ValueError("Only Mean takes a horizon qualifier")
    return SHORT[family] + ("_" + suffix if suffix else "")


def resolve_visual_arm(name):
    """Visual composition or qualified history-control arm, never a new route."""
    if name.lower().startswith(("pc-", "pc_")):
        key = name.lower().replace("-", "_")
        if "_pretrained_" in key or "_random_" in key:
            return resolve_control(name)
    return resolve_method(name, "visual")


def resolve_coupled_arm(name):
    """Keep the Frozen-only Coupled contract; accept an explicit Frozen suffix."""
    if not name.lower().startswith(("pc-", "pc_")):
        return name
    name, sep, route = name.partition("__")
    if sep and route != "Frozen":
        raise ValueError("The published Coupled API is Frozen-only")
    return resolve_method(name, "coupled") + ("__Frozen" if sep else "")


def resolve_coupled_ground(name):
    """Resolve a complete ground arm; public compositions imply Frozen only."""
    if name is None: raise ValueError("ground requires an arm")
    if name.lower().startswith(("pc-", "pc_")) and "__" not in name:
        name += "__Frozen"
    return resolve_coupled_arm(name)


def normalize_config(config, interface):
    """Copy only declared method-bearing fields; do not rewrite paths or prose."""
    import copy
    result = copy.deepcopy(config)
    resolver = {"method": resolve_method, "mpe": lambda n: resolve_method(n, "mpe"),
                "control": resolve_control, "visual": resolve_visual_arm}[interface]
    fields = {"methods", "latent_methods", "arms", "fair_arms", "families",
              "control_configs", "new_control_configs", "scaling_control_configs",
              "representation_configs", "corrected_families", "full_original_arms",
              "scaling_original_arms", "probe_methods", "probe_auxiliaries",
              "primary_contrasts", "main_model",
              "secondary_model", "primary_model", "method", "model", "arm"}
    def names(value):
        if isinstance(value, str): return resolver(value)
        if isinstance(value, list): return [names(v) for v in value]
        if isinstance(value, tuple): return tuple(names(v) for v in value)
        return value
    for field in fields & result.keys(): result[field] = names(result[field])
    return result


def normalize_job(job, index, interface):
    """Normalize one named position without changing a caller's job object."""
    resolver = {"mpe": lambda n: resolve_method(n, "mpe"),
                "control": resolve_control, "visual": resolve_visual_arm}[interface]
    values = list(job)
    values[index] = resolver(values[index])
    return tuple(values) if isinstance(job, tuple) else values


def resolve_mpe_config(config):
    return normalize_config(config, "mpe")


def display_name(name, interface="method"):
    """Human label for a recorded ID. Machine records keep the recorded ID."""
    if "__" in name:
        base, route = name.rsplit("__", 1)
        if route in {"Frozen", "Adapt"}:
            return display_name(base, interface) + " / " + route
    mapped = paper_name(name, "method" if interface == "control" else interface)
    if mapped == name: mapped = paper_name(name)
    if mapped != name: return mapped
    if name.startswith("base+"):
        short = display_name(name[5:], "control")
        return short + "-Aux" if short.startswith("PC-") else name
    if name.startswith("anchor_") and name[7:] in {"mif", "simple", "graph", "tree", "edge_h1", "edge_h2", "edge_h3"}:
        return display_name(name[7:], "control") + "-Aux"
    for public, short in SHORT.items():
        if name == short and public != "pc_mean": return LABELS[public]
        if name.startswith(short + "_"):
            suffix = name[len(short)+1:]
            if CONTROL_SUFFIX.fullmatch(suffix):
                return LABELS[public] + "-" + "-".join({"solo": "Solo", "aux": "Aux"}.get(v, v) for v in suffix.split("_"))
    return name


def report_text(lines, interface="method"):
    """Format Markdown displays only; never edit JSON evidence or source paths.

    Identifiers in table cells/bullets get paper names. Paths, filenames and
    inline code remain literal for traceability; MTE mathematical prose stays.
    """
    if interface not in {"method", "mpe", "control"}: raise ValueError(interface)
    output = []
    for line in lines:
        parts = re.split(r"(`[^`]*`|(?=[^\s|]*[.\\])[A-Za-z0-9_.:+-]*[/\\][^\s|]*|[A-Za-z0-9_]+\.(?:py|json|md|pt|npz))", line)
        for i in range(0, len(parts), 2):
            part = parts[i]
            if line.startswith(("|", "- ")):
                part = re.sub(r"[A-Za-z][A-Za-z0-9_+]*(?:__Frozen|__Adapt)?",
                              lambda m: display_name(m.group(), interface), part)
            for old, new in [("Edge-CARA", "PC-Mean"), ("MIF", "PC-Field"), ("Simple", "PC-Set"), ("Graph", "PC-Graph"), ("Tree", "PC-Sweep")]:
                part = re.sub(r"(?<![A-Za-z0-9_-])" + old + r"(?![A-Za-z0-9_])", new, part)
            parts[i] = part
        output.append("".join(parts))
    return "\n".join(output)


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve")
    parser.add_argument("--interface", choices=["method", "visual", "coupled", "mpe", "control"], default="method")
    args = parser.parse_args()
    if args.resolve:
        resolver = {"control": resolve_control, "visual": resolve_visual_arm, "coupled": resolve_coupled_arm}.get(args.interface)
        print(resolver(args.resolve) if resolver else resolve_method(args.resolve, args.interface))
    else:
        print(json.dumps({LABELS[k]: v for k, v in FAMILY.items()}, indent=2))

if __name__ == "__main__":
    main()
