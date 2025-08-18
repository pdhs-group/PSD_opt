# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:05:01 2025

@author: px2030
"""

import libcst as cst
from pathlib import Path

class ReplaceConfig(cst.CSTTransformer):
    def __init__(self, key: str, new_value: str):
        self.key = key
        self.new_value = new_value

    def leave_Dict(self, original_node, updated_node):
        new_elements = []
        for elt in updated_node.elements:
            if (isinstance(elt.key, cst.SimpleString)
                and elt.key.evaluated_value == self.key):
                new_val = cst.SimpleString(f"'{self.new_value}'")
                new_elements.append(elt.with_changes(value=new_val))
            else:
                new_elements.append(elt)
        return updated_node.with_changes(elements=new_elements)

def replace_key_value(path: str, key: str, new_value: str):
    code = Path(path).read_text(encoding="utf-8")
    tree = cst.parse_module(code)
    new_tree = tree.visit(ReplaceConfig(key, new_value))
    Path(path).write_text(new_tree.code, encoding="utf-8")