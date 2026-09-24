import ast,math,torch
from torch import nn
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]/"third_party/laom/src"
HERE=Path(__file__).resolve().parent
NAMES={"MLPBlock","LatentActHead","LatentObsHead","ResidualBlock","EncoderBlock","DecoderBlock","Actor","ActionDecoder","IDM","FDM","LAPO","LAOM"}

# Final implementation source: visual_multiagent_2026_09_10/pixel_pipeline_v1/official.py:14

def load():
    util = ast.parse((SOURCE/'utils.py').read_text(encoding='utf-8'))
    source = ast.parse((SOURCE/'nn.py').read_text(encoding='utf-8'))
    nodes = [x for x in util.body if isinstance(x, ast.FunctionDef) and x.name == 'weight_init']
    nodes += [x for x in source.body if isinstance(x, ast.ClassDef) and x.name in NAMES]
    assert len(nodes) == len(NAMES) + 1
    context = dict(math=math, torch=torch, nn=nn, __name__=__name__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE/'nn.py'), 'exec'), context)
    return context

# Final implementation source: visual_multiagent_2026_09_10/pixel_pipeline_v1/official.py:25

def load_augmenter():
    # Load this standalone official file without modifying package import paths.
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('pixel_pipeline_official_augmentations', SOURCE/'augmentations.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Augmenter(64)
