# code_map.py — kaynak kod yapısını çıkarır (hafif statik analiz)
import ast, os, json
from typing import Dict, Any, List

def map_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
    except Exception as e:
        return {"file": path, "error": str(e), "functions": [], "classes": []}

    funcs, classes, imports = [], [], []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            funcs.append({
                "name": node.name,
                "args": args,
                "lineno": node.lineno,
                "doc": ast.get_docstring(node) or ""
            })
        elif isinstance(node, ast.ClassDef):
            methods = []
            for b in node.body:
                if isinstance(b, ast.FunctionDef):
                    methods.append({"name": b.name, "args": [a.arg for a in b.args.args], "lineno": b.lineno})
            classes.append({"name": node.name, "methods": methods, "lineno": node.lineno})
        elif isinstance(node, ast.Import):
            imports += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports.append(f"{node.module or ''}")

    return {"file": path, "imports": imports, "functions": funcs, "classes": classes}

def map_project(root: str = ".") -> Dict[str, Any]:
    pyfiles: List[str] = []
    for r,_,files in os.walk(root):
        for fn in files:
            if fn.endswith(".py") and not fn.startswith("."):
                pyfiles.append(os.path.join(r, fn))
    out = [map_file(p) for p in pyfiles]
    with open("code_map.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return {"files": len(pyfiles), "output": "code_map.json"}
