# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:05:01 2025

@author: px2030
"""

import ast
from pathlib import Path

def replace_value_for_key_in_py(path: str, key: str, new_value: str) -> int:
    code = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(code, filename=path)

    class Replacer(ast.NodeTransformer):
        def __init__(self, key, new_value):
            self.key = key
            self.new_node = ast.Constant(value=new_value)
            self.count = 0

        def visit_Dict(self, node: ast.Dict):
            for i, k in enumerate(node.keys):
                if isinstance(k, ast.Constant) and k.value == self.key:
                    node.values[i] = self.new_node
                    self.count += 1
            self.generic_visit(node)
            return node

    tr = Replacer(key, new_value)
    new_tree = tr.visit(tree)
    ast.fix_missing_locations(new_tree)

    new_code = ast.unparse(new_tree)
    Path(path).write_text(new_code, encoding="utf-8")
    return tr.count