"""Verify scientific core bodies against AST fingerprints of executed sources."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class CoreFingerprintTests(unittest.TestCase):
    def test_executed_scientific_functions_preserved(self):
        expected=json.loads((ROOT/'configs/core_function_sha256.json').read_text(encoding='utf-8'))
        for relative,functions in expected.items():
            nodes={node.name:node for node in ast.parse((ROOT/relative).read_text(encoding='utf-8-sig')).body
                   if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
            for name,digest in functions.items():
                with self.subTest(file=relative,function=name):
                    actual=hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest()
                    self.assertEqual(actual,digest)


if __name__=='__main__':unittest.main()
