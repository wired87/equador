import time
import json
from typing import Any, List, Dict

from firegraph.equador.eq_extractor import EqExtractor
from firegraph.equador.module_loader import ModuleLoader
from firegraph.qfu.field_utils import FieldUtils



class Modulator(
    ModuleLoader,
    EqExtractor,
):
    def __init__(
            self,
            g,
            mid: str,
            qfu,
    ):
        """
        Initialize the Modulator instance state.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `G`, `mid`, `qfu`.
        2. Delegates side effects or helper work through `GUtils()`, `FieldUtils()`, `EqExtractor.__init__()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `G`: Graph instance that the workflow reads from or mutates.
        - `mid`: Caller-supplied value used during processing. Expected type: `str`.
        - `qfu`: Caller-supplied value used during processing.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        self.id = mid
        self.g = g
        self.qfu = qfu
        self.fu = FieldUtils()
        
        # Initialize StructInspector manually as ModuleLoader init is skipped
        self.current_class = None
        
        #StateHandler.__init__(self)
        EqExtractor.__init__(self, self.g)


    def register_modules_G(self, modules: List[Dict]):
        """
        Registers modules and their parameters into the local graph (self.g).
        """
        print(f"Registering {len(modules)} modules to G")
        try:
            for module in modules:
                module_id = module.get("id")
                if not module_id:
                    print(f"Skipping module without id: {module}")
                    continue

                # 1. Add Module Node
                # Ensure nid and type are set for GUtils.add_node
                module_attrs = module.copy()
                module_attrs["id"] = module_id
                module_attrs["type"] = "MODULE"
                
                self.g.add_node(attrs=module_attrs)

                # 2. Process Params
                params_raw = module.get("params")
                params = {}
                if params_raw:
                    if isinstance(params_raw, str):
                        try:
                            params = json.loads(params_raw)
                        except Exception as e:
                            print(f"Error parsing params for module {module_id}: {e}")
                            params = {}
                    elif isinstance(params_raw, dict):
                        params = params_raw
                
                # 3. Create Param Nodes and Links
                for param_id, data_type in params.items():
                    # Add Param Node
                    param_attrs = {
                        "id": param_id,
                        "type": "PARAM",
                        "data_type": data_type
                    }
                    self.g.add_node(attrs=param_attrs)
                    
                    # Link Module -> Param
                    edge_attrs = {
                        "rel": "uses_param",
                        "src_layer": "MODULE",
                        "trgt_layer": "PARAM"
                    }
                    self.g.add_edge(
                        src=module_id,
                        trgt=param_id,
                        attrs=edge_attrs
                    )

        except Exception as e:
            print(f"Err core.qbrain.core.module_manager.modulator::Modulator.register_modules_G | handler_line=105 | {type(e).__name__}: {e}")
            print(f"[exception] core.qbrain.core.module_manager.modulator.Modulator.register_modules_G: {e}")
            print(f"Error in register_modules_G: {e}")


    def set_constants(self):
        """
        Set constants for the Modulator workflow.

        Workflow:
        1. Starts from the current object state and local workflow context.
        2. Builds intermediate state such as `env_keys` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `print()`, `list()`, `self.g.G.nodes()`.
        5. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - None.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        print("set_constants...")
        env_keys = list(self.qfu.create_env().keys())
        for k, v in self.g.G.nodes(data=True):
            if v.get("type") != "PARAM":
                continue
            if k in env_keys:
                v["const"] = True
            else:
                v["const"] = False
        print("set_constants... done")



    def module_conversion_process(self):
        print("module_conversion_process...")
        try:
            ModuleLoader.__init__(
                self,
                g=self.g,
                id=self.id,
            )

            self.load_local_module_codebase(
                code_base=self.g.get_node(self.id).get("code")
            )
            print("load_local_module_codebase... ")
            
            # code -> G
            self.create_code_G(mid=self.id)
            print("create_code_G... ")

            # G -> sorted runnables -> add inde to node
            self.set_constants()

            print("module_conversion_process... done")
        except Exception as e:
            print("MODULE CONVERSION FAILED:", e)



    def set_field_data(self, field, dim=3):
        """
        Set example field data
        """
        print("set_field_data for ", field)
        try:

            data: dict = self.qfu.batch_field_single(
                ntype=field,
                dim=dim,
            )

            # set params for module
            keys = list(data.keys())
            values = list(data.values())

            axis_def = self.set_axis(values)
            print(f"update field node {field}")
            self.g.update_node(
                dict(
                    id=field,
                    keys=keys,
                    values=values,
                    axis_def=axis_def,
                )
            )
        except Exception as e:
            print(f"Err core.qbrain.core.module_manager.modulator::Modulator.set_field_data | handler_line=214 | {type(e).__name__}: {e}")
            print(f"[exception] core.qbrain.core.module_manager.modulator.Modulator.set_field_data: {e}")
            print("Err set_field_data:", e)
        print("create_modules finished")


    def set_pattern(self):
        """
        Set pattern for the Modulator workflow.

        Workflow:
        1. Starts from the current object state and local workflow context.
        2. Builds intermediate state such as `STRUCT_SCHEMA`, `node`, `keys` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `keys.index()`, `self.g.get_neighbor_list()`, `self.fu.env.index()`.
        5. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - None.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        STRUCT_SCHEMA = []

        for f in self.fields:
            node = self.g.G.nodes[f]
            keys = node["keys"]

            # loop single eqs
            for struct in self.arsenal_struct:
                for p in struct["params"]:
                    struct_item = []
                    # param lccal?
                    if p in keys:
                        struct_item = [
                            self.module_index,
                             node["field_index"],
                            keys.index(p)
                        ]

                    elif p in self.fu.env:
                        struct_item = [
                            self.module_index,
                            [0], # first and single field
                            self.fu.env.index(p),
                        ]

                    else:
                        # param from neighbor field ->
                        # get all NEIGHBOR FIELDS
                        nfs = self.g.get_neighbor_list(
                            node=f["id"],
                            target_type="FIELD",
                        )

                    STRUCT_SCHEMA[
                        node["field_index"]
                    ] = struct_item







    def set_return_des(self):
        # param: default v
        """
        Set return des for the Modulator workflow.

        Workflow:
        1. Starts from the current object state and local workflow context.
        2. Builds intermediate state such as `field_param_map`, `k`, `return_param_pattern` before applying the main logic.
        3. Iterates through the available items and applies the same workflow rules consistently.
        4. Delegates side effects or helper work through `self.g.G.nodes[self.id]['field_param_map'].keys()`, `list()`, `enumerate()`.
        5. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - None.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        field_param_map: dict[str, Any] = self.g.G.nodes[self.id]["field_param_map"].keys()

        # create PATTERN
        k = list(field_param_map.keys())
        return_param_pattern = [
            None
            for _ in range(len(k))
        ]

        # LINKFIELD PARAM -> RETURN KEY
        for i, item in enumerate(self.arsenal_struct):
            return_key = item["return_key"]
            return_param_pattern[i]: int = field_param_map.index(return_key)
        print(f"{self.id} runnable creared")



    def create_field_workers(
            self,
            fields:list[str]
    ):
        # todo may add module based axis def
        """
        Create field workers for the Modulator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `fields`.
        2. Builds intermediate state such as `start`, `end` before applying the main logic.
        3. Iterates through the available items and applies the same workflow rules consistently.
        4. Delegates side effects or helper work through `time.perf_counter_ns()`, `print()`, `enumerate()`.
        5. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `fields`: Caller-supplied value used during processing. Expected type: `list[str]`.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        start = time.perf_counter_ns()
        try:
            for i, fid in enumerate(fields):
                self.set_field_data(
                    field=fid,
                )
        except Exception as e:
            print(f"Err core.qbrain.core.module_manager.modulator::Modulator.create_field_workers | handler_line=342 | {type(e).__name__}: {e}")
            print(f"[exception] core.qbrain.core.module_manager.modulator.Modulator.create_field_workers: {e}")
            print("Err create_field_workers", e)
        end = time.perf_counter_ns()
        print("Field Workers created successfully after s:", end - start)

    def set_axis(self, data:list) -> tuple:
        """
        Determines the vmap axis for each parameter in the admin_data bundle.
        - Use axis 0 for array-like admin_data (map over it).
        - Use None for scalar admin_data (broadcast it).
        """
        return tuple(
            0
            if not isinstance(
                param, (int, float)
            )
            else None
            for param in data
        )

