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
    key = name.lower().replace("-", "_")
    match = re.fullmatch(r"(pc_mean|pc_set|pc_graph|pc_sweep|pc_field)(_h[123])?(_solo|_aux)?", key)
    if not match:
        raise ValueError("Unknown predictive-comparison name: " + name)
    family, horizon, composition = match.groups()
    if interface == "mpe":
        names = {"pc_set": "simple", "pc_graph": "graph", "pc_sweep": "tree", "pc_field": "mif"}
        if family == "pc_mean":
            if horizon not in {"_h2", "_h3"}:
                raise ValueError("MPE Mean requires its explicit h2 or h3 protocol")
            native = "edge" + horizon
        else:
            if horizon:
                raise ValueError("Only Mean takes an explicit MPE horizon")
            native = names[family]
        return ("base+" if composition == "_aux" else "") + native
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
    if interface != "method":
        raise ValueError("Unknown display interface: " + interface)
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
    return name + route if route else name

def resolve_mpe_config(config):
    """Copy a caller's config and normalize names; preserve budgets and input data."""
    import copy
    result = copy.deepcopy(config)
    for field in ("methods", "latent_methods"):
        if field in result:
            result[field] = [resolve_method(name, "mpe") for name in result[field]]
    return result

def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve")
    parser.add_argument("--interface", choices=["method", "visual", "coupled", "mpe"], default="method")
    args = parser.parse_args()
    if args.resolve:
        print(resolve_method(args.resolve, args.interface))
    else:
        print(json.dumps({LABELS[k]: v for k, v in FAMILY.items()}, indent=2))

if __name__ == "__main__":
    main()
