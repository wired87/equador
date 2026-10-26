import ast
import importlib
from typing import Union, Optional, Callable

from qbrain.graph.local_graph_utils import GUtils

def _get_docstring(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> str:
    """Extracts docstring from function or class node."""
    return ast.get_docstring(node) or ""


def _get_type_name(node: Optional[ast.expr]) -> str:
    """Extracts type name from annotation or defaults to 'Any'."""
    return ast.unparse(node) if node else 'Any'


def _has_self_param(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
    """True if function has 'self' as first parameter (i.e. is a class method)."""
    if node.args.args:
        return node.args.args[0].arg == "self"
    return False


def _make_direct_callable(
    module_name: str,
    method_name: str,
    class_name: Optional[str] = None,
    has_self: bool = False,
) -> Callable:
    """
    Create a callable that invokes the function/method without requiring the caller
    to pass self or a class instance. Accepts payload (dict) and flattens to **kwargs
    for handler methods. For class methods: instantiates the class (no-arg __init__).
    """
    def invoke(*args, **kwargs):
        from qbrain.core.handler_utils import flatten_payload
        mod = importlib.import_module(module_name)
        if class_name and has_self:
            cls = getattr(mod, class_name)
            instance = cls()
            func = getattr(instance, method_name)
            if args and len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                return func(**flatten_payload(args[0]))
            return func(*args, **kwargs)
        else:
            func = getattr(mod, method_name)
            if args and len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                return func(**flatten_payload(args[0]))
            return func(*args, **kwargs)
    return invoke




class EqInspector(ast.NodeVisitor):

    """
    USE FOR SINGLE FILE
    Traverses AST to populate a CodeGraph with classes, methods, and variables.
    The graph stores the entire structure; no redundant internal dicts are kept.
    """

    def __init__(self, G=None):
        """
        Initialize the StructInspector instance state.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `G`.
        2. Delegates side effects or helper work through `GUtils()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `G`: Graph instance that the workflow reads from or mutates.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        self.current_class: Optional[str] = None
        self.g:GUtils = GUtils(G=G)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Track current class for method resolution."""
        prev = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev

    # B. Visit Methods (Sync/Async)
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit FunctionDef for the StructInspector workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `node`.
        2. Delegates side effects or helper work through `self._process_function()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `node`: Caller-supplied value used during processing. Expected type: `ast.FunctionDef`.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """
        Visit AsyncFunctionDef for the StructInspector workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `node`.
        2. Delegates side effects or helper work through `self._process_function()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `node`: Caller-supplied value used during processing. Expected type: `ast.AsyncFunctionDef`.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        self._process_function(node)

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        """Processes methods (or standalone functions, if not in class).
        Creates a direct callable for each function/method that can be invoked
        without passing self or class instance (for methods, instantiates internally).
        """
        try:
            method_name = node.name
            has_self = _has_self_param(node)
            method_id = f"{self.current_class}.{method_name}" if self.current_class else method_name
            print(f"CREATE METHOD:{self.module_name}:", method_id)

            if not method_name.startswith("_"):

                print("Get Method Data")
                return_type = _get_type_name(node.returns)
                docstring = _get_docstring(node)

                return_key = self.extract_return_statement_expression(
                    method_node=node,
                )

                print(f"RETURN KEY FOR METHOD {method_name}", return_key)
                if len(return_key.split(" ")) == 0:
                    return_key = method_id

                entire_def = ast.unparse(node)

                callable_fn = _make_direct_callable(
                    module_name=self.module_name,
                    method_name=method_name,
                    class_name=self.current_class,
                    has_self=has_self,
                )

                #p_tree = _extract_param_tree(node)
                #p_tree_str = json.dumps(p_tree) if p_tree else "{}"
                # print("METHOD-module edge created", method_id)
                params = self.process_method_params(node, method_id, return_key)

                data = {
                    "id": method_id,
                    "tid": 0,
                    "parent": ["MODULE"],
                    "type": "METHOD",
                    "return_key": return_key,
                    "returns": return_type,
                    "docstring": docstring,
                    "code": entire_def,
                    "module_id": self.module_name,
                    "callable": callable_fn,
                    "params": params,
                }

                # 1. METHOD Node
                self.g.add_node(
                    attrs=data
                )

                print("METHOD node created", method_id)

                # MODULE -> METHOD
                self.g.add_edge(
                    src=self.module_name,
                    trgt=method_id,
                    attrs=dict(
                        rel='has_method',
                        trgt_layer='METHOD',
                        src_layer='MODULE',
                    )
                )

        except Exception as e:
            print("Err _process_function", e)
        self.generic_visit(node)


    def process_method_params(self, node, method_id, return_key):
        filtered_args = []
        return_key = return_key.strip()

        for arg in node.args.args:
            try:
                if arg.arg == 'self': continue
                param_name = arg.arg
                param_type = _get_type_name(arg.annotation)

                print("ADD PARAM:", param_name)
                # PARAM
                self.g.add_node(
                    attrs=dict(
                        id=param_name,
                        type='PARAM',
                        param_type=param_type,
                    )
                )

                # METHOD -> PARAM
                self.g.add_edge(
                    src=method_id,
                    trgt=param_name,
                    attrs=dict(
                        rel='requires_param',
                        type=param_type,
                        trgt_layer='PARAM',
                        src_layer='METHOD',
                    ))

                # MODULE -> PARAM
                if not self.g.G.has_edge(self.module_name, param_name):
                    self.g.add_edge(
                        src=self.module_name,
                        trgt=param_name,
                        attrs=dict(
                            rel='requires_param',
                            trgt_layer='PARAM',
                            src_layer='MODULE',
                        ))

                filtered_args.append(arg.arg)
            except Exception as e:
                print(f"Err core.qbrain.code_manipulation.graph_creator::StructInspector.process_method_params | handler_line=263 | {type(e).__name__}: {e}")
                print(f"[exception] core.qbrain.code_manipulation.graph_creator.StructInspector.process_method_params: {e}")
                print("Err node.args.args", e)

            # RETURN PARAM
            self.g.add_node(
                attrs=dict(
                    id=return_key,
                    type='PARAM',
                )
            )
            # METHOD -> PARAM
            self.g.add_edge(
                src=method_id,
                trgt=return_key,
                attrs=dict(
                    rel='returns_param',
                    trgt_layer='PARAM',
                    src_layer='METHOD',
                ))

            # MODULE -> PARAM
            if not self.g.G.has_edge(
                    self.module_name,
                    return_key
            ):
                self.g.add_edge(
                    src=self.module_name,
                    trgt=return_key.strip(),
                    attrs=dict(
                        rel='returns_param',
                        trgt_layer='PARAM',
                        src_layer='MODULE',
                    ))
        print("process_method_params... done -> filtered_args", filtered_args)



    def extract_return_statement_expression(self, method_node: ast.FunctionDef) -> Optional[str]:
        """
        Extracts the name/identifier of the returned expression, stripped of whitespace.
        """
        for node in ast.walk(method_node):
            try:
                if isinstance(node, ast.Return) and node.value is not None:
                    # ast.unparse converts the AST node back into a string
                    # .strip() removes any leading/trailing whitespace or newlines
                    return ast.unparse(node.value).strip()
            except Exception as e:
                # It's often better to log this or use 'pass' if you want it silent
                print(f"Err core.qbrain.code_manipulation.graph_creator::StructInspector.extract_return_statement_expression | handler_line=311 | {type(e).__name__}: {e}")
                print(f"[exception] core.qbrain.code_manipulation.graph_creator.StructInspector.extract_return_statement_expression: {e}")
                print(f"Err extract_return_statement_expression: {e}")
        return ""



    # C. Visit Class Variables
    def visit_Assign(self, node: ast.Assign):
        """Identifies class variables and creates CLASS_VAR nodes."""

        if not self.current_class: return

        for target in node.targets:
            try:
                if isinstance(target, ast.Name):
                    var_name = target.id

                    value_type = 'Unknown'
                    var_id = f"{self.current_class}.{var_name}"

                    # 1. Add CLASS_VAR Node
                    self.g.add_node(
                        dict(
                            id=var_id,
                            type='CLASS_VAR',
                            name=var_name,
                            inferred_type=value_type,
                        )
                    )
            except Exception as e:
                print(f"Err core.qbrain.code_manipulation.graph_creator::StructInspector.visit_Assign | handler_line=343 | {type(e).__name__}: {e}")
                print(f"[exception] core.qbrain.code_manipulation.graph_creator.StructInspector.visit_Assign: {e}")
                print("Err Agign ", e)

    def convert_module_to_graph(self, code_content:str, module_name):
        """
        Parses code content, runs the inspector, and returns the graph admin_data.
        Takes code as a string input, as required by ast.parse.
        """
        self.module_name = module_name
        print("ADD MODULE:", module_name)
        try:
            # Check if code is empty
            if not code_content.strip():
                return {
                    "Error": "Input code content is empty."
                }

            tree = ast.parse(code_content)
            self.visit(tree)

        except Exception as e:
            print(f"Err {type(e).__name__}: {e}")
            


