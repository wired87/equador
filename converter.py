from __future__ import annotations

import base64
import os
import pprint

import json
import numpy as np

from firegraph.code.graph_creator import StructInspector

from firegraph.equador.mcreator import ModuleCreator
from firegraph.qfu.all_subs import ALL_SUBS
from firegraph.qfu.qf_utils import QFUtils
from firegraph.utils._str import clean_underscores_front_back, rm_prev_mark
from firegraph.utils.get_shape import extract_complex
from firegraph.equador.eq_extractor import EqExtractor
from firegraph.utils._np.expand_array import expand_structure
from firegraph.visual import create_g_visual

EXCLUDED_ORIGINS = ["neighbor", "interactant"]


# gien: prompt — gate hot-loop debug prints so Guard graph build stays fast unless COR_VERBOSE=1
def _cor_verbose() -> bool:
    return (os.environ.get("COR_VERBOSE") or "0").strip().lower() in ("1", "true", "yes", "on")


class Guard:

    def __init__(
        self,
        amount_nodes,
        sim_time,
        dims,
        qfu,
        g,
        user_id="public",
        cfg_file="sim_config.json"
    ):
        print("Initializing Guard...")
        self.user_id = user_id

        self.amount_nodes = amount_nodes
        self.sim_time = sim_time
        self.dims = dims

        self.world_cfg = None

        self.qfu:QFUtils = qfu
        self.g = g
        self.ready_map = {
            k: False
            for k in ALL_SUBS
        }

        self.testing = True

        self.mcreator = ModuleCreator(
            self.g,
            self.qfu,
        )

        self.eq_extractor = EqExtractor()

        self.code_extractor = StructInspector(
            g=self.g,
        )

        self.cfg_file = cfg_file
        # gien: Guard lives in repo root — use that folder (not parent of repo) for sim JSON
        _repo_root = os.path.dirname(os.path.abspath(__file__))
        # gien: avoid `sim_config.json.json` when caller passes a full filename
        _cfg_name = cfg_file if str(cfg_file).lower().endswith(".json") else f"{cfg_file}.json"
        self.cfg_file = os.path.join(_repo_root, _cfg_name)

        self.fields = []
        # gien: flat output root beside repo — runtime.json lands here from Guard.main (not JaxGuard)
        self._out_root = os.path.join(_repo_root, "output")
        print("Guard Initialized!")

    
    def _export_runtime(self, cfg: dict) -> None:
        # gien: prompt — persist runtime specs from class args + main(env_id), never from os.environ
        try:
            # SAVE CFG
            os.makedirs(self._out_root, exist_ok=True)
            _path = os.path.join(self._out_root, "runtime.json")
            
            with open(_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            print("runtime.json saved →", _path)

            # SAVE G VIS
            vis_path= os.path.join(self._out_root, "visuals.html")
            create_g_visual(self.g.G, dest_path=vis_path, ds=False)

            # GRAPH SAVE
            graph_path= os.path.join(self._out_root, "graph.json")
            self.g.save_graph(
                dest_file=graph_path,
            )
        except Exception as exc:
            print(f"Warn _export_runtime: {type(exc).__name__}: {exc}")


    def _ensure_ghost_module(self, env_id: str) -> None:
        """SMManager graphs never run ComponentGraphCreator; iterators still need GHOST_MODULE."""
        if self.g.G.has_node("GHOST_MODULE"):
            return
        #env = self.g.get_node(env_id)
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
            just_id=True
        )

        self.g.add_node(dict(
            id="GHOST_MODULE",
            type="MODULE",
            module_index=len(modules),

        ))

        self.g.add_edge(
            src="GHOST_MODULE",
            trgt=env_id,
            attrs=dict(
                rel="has_field",
                src_layer="ENV",
                trgt_layer="MODULE",
            ))
        print("ghost od created")

    def _normalize_module_indices(self) -> None:
        """Contiguous module_index on real MODULEs; ghost last — keeps DB / iterators aligned."""
        modules = [
            mid
            for mid, attrs in self.g.G.nodes(data=True)
            if (attrs or {}).get("type") == "MODULE" and "GHOST" not in str(mid).upper()
        ]
        modules.sort()
        for i, mid in enumerate(modules):
            self.g.update_node({"id": mid, "module_index": i})
        if self.g.G.has_node("GHOST_MODULE"):
            self.g.update_node({"id": "GHOST_MODULE", "module_index": len(modules)})


    def main(self, env_id="public"):
        # CREATE COMPONENT CFG ASND SAVE IT IN ROOT
        print("Guard.main...")
        components = self.converter(
            env_id
        )

        #
        self._export_runtime(
            components,
        )
        
        #
        return components

    def handle_deployment(self, env_id, components):
        print(f"\n[START] handle_deployment für Env: {env_id}")

        try:
            if not self._is_components_valid_for_grid(components):
                print("  [WARN] Aborting deployment: components have empty/invalid structures for grid-root")
                return
            
            # 1. Config Erstellung
            world_cfg = self.create_vm_cfgs(env_id)
            print(f"  -> world_cfg erstellt:", type(world_cfg))
            pprint.pp(world_cfg)

            # 2–3. Local-only: no remote DB pattern update; env vars for optional local runners only
            container_env: dict = self.deployment_handler.env_creator.create_env_variables(
                env_id=env_id,
                cfg=world_cfg
            )
            print("  -> Container env dict (local keys only):", len(container_env), "entries")

            print(f"  -> Modus: local (cfg file only, no cloud DB status)")
            import os
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_out.json")
            with open(path, "w") as f:
                f.write(json.dumps(components, indent=4))
            print(f"  -> Datei erfolgreich geschrieben unter: {path}")

        except Exception as e:
            print(f"Err in handle_deployment: {e}")




    def _push_grid_frame(self, components: dict, grid_streamer, step: int = 0) -> None:
        """Decode DB from components and push to grid streamer (non-blocking)."""
        db_b64 = components.get("DB")
        if not db_b64:
            return
        try:
            raw = base64.b64decode(db_b64)
            arr = np.frombuffer(raw, dtype=np.complex64)
            data = np.abs(arr).astype(np.float32) if np.iscomplexobj(arr) else arr.astype(np.float32)
            grid_streamer.put_frame(step, data)
        except Exception as e:
            print(f"[guard] _push_grid_frame: {e}")

    def _save_animation_frame(self, components: dict, recorder, step: int = 0) -> None:
        """Decode DB and save plot frame for animation recorder."""
        db_b64 = components.get("DB")
        if not db_b64:
            return
        try:
            raw = base64.b64decode(db_b64)
            arr = np.frombuffer(raw, dtype=np.complex64)
            data = np.abs(arr).astype(np.float32) if np.iscomplexobj(arr) else arr.astype(np.float32)
            recorder.cfg = components
            recorder.save_frame(step, data)
        except Exception as e:
            print(f"[guard] _save_animation_frame: {e}")

    def _save_model_path_to_envs(self, env_id: str, model_path: str) -> None:
        """Local-only: log artifact paths (no DB)."""
        try:
            model_path_abs = os.path.abspath(model_path)
            npz_path = model_path_abs.replace(".json", "_data.npz")
            extra = f", npz={npz_path}" if os.path.isfile(npz_path) else ""
            print(f"[guard] model artifacts (local): env_id={env_id} model={model_path_abs}{extra}")
        except Exception as e:
            print(f"[guard] _save_model_path_to_envs error: {e}")



    def converter(self, env_id:str, ):
        print("Main started...")
        env_node = self.g.get_node(env_id)

        if not env_node:
            print("Err env_node None")
            return

        self._ensure_ghost_module(env_id)
        self._normalize_module_indices()

        # include gm
        modules: list = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        #
        self.DB = self.create_db(modules)

        # create iterator last
        iterators = self.set_iterator_from_humans()

        # 
        db_to_method_struct:dict = self.set_edge_db_to_method(modules, env_id)

        # 
        method_struct:dict = self.method_layer(
            self.g.get_nodes( 
                filter_key="type",
                filter_value="MODULE",
            ),
        )

        #
        method_to_db = self.set_edge_method_to_db()

        #
        #injection_patterns = self.handle_energy_components(env_node)

        components = {
            **self.DB,
            **iterators,
            **method_struct,
            #**injection_patterns,
            **db_to_method_struct,
            "METHOD_TO_DB": method_to_db or [],
            "AMOUNT_NODES": self.amount_nodes,
            "SIM_TIME": self.sim_time,
            "DIMS": self.dims,
        }

        components = self._sanitize_components(components)
        self._validate_components_no_empty(components)
        print("Main... done")
        return components

    def _sanitize_components(self, components: dict) -> dict:
        """Ensure no None or invalid empty structures that would break grid-root."""
        out = dict(components)
        list_keys = [
            "DB", "AXIS", "DB_SHAPE", "AMOUNT_PARAMS_PER_FIELD", "DB_PARAM_CONTROLLER",
            "DB_KEYS", "MODULES", "FIELDS",
            "DB_TO_METHOD_EDGES", "METHOD_TO_DB", "DB_CTL_VARIATION_LEN_PER_EQUATION",
            "DB_CTL_VARIATION_LEN_PER_FIELD", "LEN_FEATURES_PER_EQ",
            "METHOD_PARAM_LEN_CTLR", "METHODS_PER_MOD_LEN_CTLR", "NEIGHBOR_CTLR",
            "INJECTOR_TIME", "INJECTOR_INDICES", "INJECTOR_VALUES",
            "E_KEY_MAP_PER_FIELD",
        ]
        for k in list_keys:
            if k in out and out[k] is None:
                out[k] = []
        if "METHOD_TO_DB" in out and not isinstance(out["METHOD_TO_DB"], list):
            out["METHOD_TO_DB"] = []

        if "DB_TO_METHOD_EDGES" in out and not isinstance(out["DB_TO_METHOD_EDGES"], list):
            out["DB_TO_METHOD_EDGES"] = []
        return out

    _OPTIONAL_EMPTY_KEYS = frozenset({
        "INJECTOR_TIME", "INJECTOR_INDICES", "INJECTOR_VALUES",
        "E_KEY_MAP_PER_FIELD",
    })

    def _validate_components_no_empty(self, components: dict, strict: bool = True) -> None:
        """Ensure no cfg entry in components is empty. Raises ValueError with details when strict.
        Keys in _OPTIONAL_EMPTY_KEYS may be empty (e.g. when no injections).
        When strict=False (e.g. LOCAL_DB), logs warning instead of raising."""
        if not components:
            if strict:
                raise ValueError("[guard] components dict is empty")
            print("[guard] WARN: components dict is empty")
            return
        empty_keys = []
        for k, v in components.items():
            if k in self._OPTIONAL_EMPTY_KEYS:
                continue
            if v is None:
                empty_keys.append(f"{k}=None")
            elif isinstance(v, (list, dict)) and len(v) == 0:
                empty_keys.append(f"{k}=[]/{{}}")
            elif isinstance(v, str) and v.strip() == "":
                empty_keys.append(f"{k}=''")
        if empty_keys:
            msg = f"[guard] cfg entries must not be empty. Empty: {', '.join(empty_keys)}"
            if strict:
                raise ValueError(msg)
            print(f"[guard] WARN: {msg}")

    def _is_components_valid_for_grid(self, components: dict) -> bool:
        """Check if components have minimal non-empty structure for grid-root."""
        if not components:
            return False
        if components.get("DB") is None:
            return False
        if "AXIS" not in components or "FIELDS" not in components:
            return False
        modules = components.get("MODULES", [])
        fields = components.get("FIELDS", [])
        if not modules or not fields:
            return False
        return True


    def deploy_vms(self, vm_payload):
        """
        Deploy vms for the Guard workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `vm_payload`.
        2. Branches on validation or runtime state to choose the next workflow path.
        3. Delegates side effects or helper work through `print()`, `vm_payload.items()`, `isinstance()`.
        4. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `vm_payload`: Caller-supplied value used during processing.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        print("Deploying VMs...")
        if not vm_payload or not isinstance(vm_payload, dict):
            print("  [WARN] deploy_vms: empty or invalid vm_payload, skipping")
            return
        for key, cfg in vm_payload.items():
            if not isinstance(cfg, dict):
                print(f"  [WARN] deploy_vms: skipping invalid cfg for key={key}")
                continue
            print("deploy vm from", cfg)


            try:
                self.deployment_handler.create_instance(
                    **cfg
                )
            except Exception as e:
                print(f"Err cor.qbrain.cor.guard::Guard.deploy_vms | handler_line=1182 | {type(e).__name__}: {e}")
                print(f"[exception] cor.qbrain.cor.guard.Guard.deploy_vms: {e}")
                print(f"  [!!!] deploy_vms failed for {key}: {e}")
                import traceback
                traceback.print_exc()
                raise
        print("Deployment Finished!")


    def create_vm_cfgs(
            self,
            env_id: str,
            **cfg
        ):
        """
        Create vm cfgs for the Guard workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `env_id`, `**cfg`.
        2. Builds intermediate state such as `env_node`, `world_cfg` before applying the main logic.
        3. Delegates side effects or helper work through `self.g.get_node()`, `env_node.get()`, `env_node.items()`.
        4. Returns the assembled result to the caller.

        Inputs:
        - `env_id`: Identifier used to target the relevant entity. Expected type: `str`.
        - `**cfg`: Configuration payload consumed by the workflow.

        Returns:
        - Returns the computed result for the caller.
        """
        env_node = self.g.get_node(env_id)
        env_node = {k:v for k,v in env_node.items() if k not in ["updated_at", "created_at"]}
        #print("add env node to world cfg:", env_node)

        # BOB BUILDER ACTION
        world_cfg = {
            **cfg,
            "ENV_ID": env_id,
            "START_TIME": env_node.get("sim_time", 1),
            "AMOUNT_NODES": env_node.get("amount_of_nodes", 1),
            "DIMS": env_node.get("dims", 3),
        }

        return world_cfg


    def set_param_index_map(self):
        # create param index map
        """
        Set param index map for the Guard workflow.

        Workflow:
        1. Starts from the current object state and local workflow context.
        2. Builds intermediate state such as `pindex_map_fields`, `fields`, `arsenal_struct` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `self.g.get_nodes()`, `print()`, `range()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - None.

        Returns:
        - Returns the computed result for the caller.
        """
        pindex_map_fields = [
            [] for _ in range(
            len(self.fields))
        ]

        # get all modules
        fields = self.g.get_nodes(
            filter_key="type",
            filter_value="FIELD",
        )

        # todo sync finish state of all
        #  fields before
        for fid, f in fields:
            arsenal_struct = f["arsenal_struct"]
            param_map = arsenal_struct["params"]
            param_index_map = [[], []]
            for param in param_map:
                # param: str
                if param in f["keys"]:
                    # first param mapping for index
                    param_index_map[0].append(None)
                    param_index_map[1].append(
                        f["keys"].index(param)
                    )

                if fid in self.fields:
                    pindex_map_fields[
                        self.fields.index(fid)
                    ] = param_index_map

                self.g.add_node(
                    dict(
                        id=fid,
                        param_index_map=param_index_map
                    )
                )
        print("create_param_index_map finisehd:", )
        return pindex_map_fields


    def all_nodes_ready(self, trgt_types) -> bool:
        """
        All nodes ready for the Guard workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `trgt_types`.
        2. Delegates side effects or helper work through `all()`, `list()`.
        3. Returns the assembled result to the caller.

        Inputs:
        - `trgt_types`: Caller-supplied value used during processing.

        Returns:
        - Returns `bool` to the caller.
        """
        return all(
            attrs["ready"] is True
            for attrs in list(
                self.g.G.nodes[ntype]
                for ntype in trgt_types
            )
        )


    def create_db(self, modules):
        print("[create_db] start")
        import json
        db = self.get_empty_field_structure()
        axis = self.get_empty_field_structure()
        shapes = self.get_empty_field_structure()
        item_len_collection = self.get_empty_field_structure()
        param_len_collection = self.get_empty_field_structure()
        field_ids = self.get_empty_field_structure()
        db_keys = self.get_empty_field_structure()

        DB = {
            "DB": [],
            "AXIS": [],
            "DB_SHAPE": [],
            "AMOUNT_PARAMS_PER_FIELD": [],
            "DB_PARAM_CONTROLLER": [],
            "DB_KEYS": [],
        }
        for mi, (mid, m) in enumerate(modules):
            m_idx = m.get("module_index", mi)
            print(f"[create_db] module={mid}, idx={m_idx}")

            fields: dict = self.g.get_neighbor_list_rel(
                node=mid,
                trgt_rel="has_field",
            )
            print("fields loaded:", len(fields))

        try:
            for mi, (mid, m) in enumerate(modules):
                m_idx = m.get("module_index", mi)
                print(f"[create_db] module={mid}, idx={m_idx}")

                fields:dict = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                )
                print("fields loaded:", len(fields))

                for f_enum_idx, (fid, fattrs) in enumerate(fields):
                    print(f"work {m_idx}:{fid}")

                    fidx = fattrs["field_index"]
                    # keys (param IDs for this field)
                    keys = fattrs.get("keys", [])
                    values = fattrs.get("values") or []

                    # DESERIALIZE
                    if isinstance(keys, str):
                        keys = json.loads(keys) if keys else []

                    if isinstance(values, str):
                        values = json.loads(values) if values else []
                        if isinstance(values, list):
                            values = [json.loads(v) for v in values]

                    # Axis metadata belongs to the field and follows its key order.
                    xdef = fattrs.get("axis_def", [])
                    if isinstance(xdef, str):
                        xdef = json.loads(xdef) if xdef else []
                    if xdef is None:
                        xdef = []
                    if len(xdef) != len(keys):
                        raise ValueError(
                            f"Field {fid!r} has {len(keys)} keys but "
                            f"{len(xdef)} axis definitions"
                        )

                    field_shapes = fattrs["shape"]

                    len_per_param_per_field = []
                    for u in values:
                        if isinstance(u, (list, tuple)):
                            len_per_param_per_field.append(len(u))
                        else:
                            len_per_param_per_field.append(1)

                    print("m_idx:", m_idx, "fidx:", fidx, f"({fid})")
                    db[m_idx][fidx] = values
                    axis[m_idx][fidx] = xdef
                    shapes[m_idx][fidx] = field_shapes
                    item_len_collection[m_idx][fidx] = len_per_param_per_field
                    param_len_collection[m_idx][fidx ] = len(values)
                    db_keys[m_idx][fidx] = keys
                    field_ids[m_idx][fidx] = fid

                    print("vals:", values)
                    print("axis:", xdef)
                    print("shapes:", field_shapes)
                    print("item_len:", len_per_param_per_field)
                    print("param_len:", len(values))
                    print("keys:", keys)
                    print("field_id:", fid)
                    print("-" * 40)

            print("[create_db] param_len_collection:", param_len_collection)
            # len is correctly set
            for i, (m_db, m_axis, m_shape, m_len, plen_item, db_key_struct) in enumerate(
                    zip(db, axis, shapes, item_len_collection, param_len_collection, db_keys)):

                for j, (f_db, f_axis, f_shape, f_len, plen_field, db_key) in enumerate(
                        zip(m_db, m_axis, m_shape, m_len, plen_item, db_key_struct)):

                    # 1d space for all params (single dim gets later scaled to n-dim in jax_test)
                    DB["DB"].extend(f_db)

                    # axis map
                    DB["AXIS"].extend(f_axis)

                    # shapes list[tuple]
                    DB["DB_SHAPE"].extend(f_shape)

                    # int len each single param in db unscaled (e.g. 3 for 000) ffrom item_len_collection
                    # from item_len_collection
                    DB["DB_PARAM_CONTROLLER"].extend(f_len)

                    # int params does each fild has param_len_collection
                    print("plen_field", plen_field)
                    DB["AMOUNT_PARAMS_PER_FIELD"].append(plen_field)
                    DB["DB_KEYS"].extend(db_key)

            flat_db = []
            for item in DB["DB"]:
                extract_complex(item, flat_db)

            DB["DB"] = base64.b64encode(
                np.array(
                    flat_db,
                    dtype=np.complex64).tobytes()
            ).decode("utf-8")

            print("[create_db] done")
        except Exception as e:
            print("[create_db][ERROR]", e)
            import traceback
            traceback.print_exc()
            raise

        print("create_db... done")
        return DB





    def create_method_param_nodes(self, modules):
        """
        todo before: fetch and add param nodes include type
        Create Param nodes form fiedls
        Goal:
        """
        for mid, module in modules:
            # ghost does not have equation
            if "GHOST" in mid.upper(): continue
            # print("method_layer... working", mid)

            fields = self.g.get_neighbor_list_rel(
                trgt_rel="has_field",
                node=mid,
                as_dict=True,
            )
            for fid, fattrs in fields.items():
                # PARAMS from METHOD
                values = fattrs.get("fields", [])
                keys = fattrs.get("keys", [])

                for param, value in zip(keys, values):
                    # type already exists
                    self.g.update_node({
                        "id": param,
                        "type": "PARAM",
                        "value": value,
                    })

                    self._edge(fid, param, "has_param", "FIELD", "PARAM")

    def is_differnetial_equation(self, params):
        """
        Is differnetial equation for the Guard workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `params`.
        2. Builds intermediate state such as `normalized`, `has_duplicates` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `any()`, `p.replace()`, `self.has_special_params()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `params`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        normalized = [p.replace("_", "") for p in params]
        has_duplicates = any(normalized.count(norm_p) == 2 for norm_p in set(normalized) if norm_p)

        if has_duplicates and self.has_special_params(params):
            return True
        return False

    def has_special_params(self, params):
        """
        Has special params for the Guard workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `params`.
        2. Delegates side effects or helper work through `any()`, `p.endswith()`, `p.startswith()`.
        3. Returns the assembled result to the caller.

        Inputs:
        - `params`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        return any(p.endswith("__") for p in params) or any(p.startswith("_prev") or p.startswith("prev_") for p in params)


    def is_interaction_eq(self, params, modules_params:list[str], modules_return_map:list[str]):
        """
        Is interaction eq for the Guard workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `params`, `modules_params`, `modules_return_map`.
        2. Builds intermediate state such as `normalized`, `has_duplicates`, `prefixed_dublet` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `any()`, `p.replace()`, `normalized.count()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `params`: Caller-supplied value used during processing.
        - `modules_params`: Caller-supplied value used during processing. Expected type: `list[str]`.
        - `modules_return_map`: Caller-supplied value used during processing. Expected type: `list[str]`.

        Returns:
        - Returns the computed result for the caller.
        """
        normalized = [p.replace("_", "") for p in params]
        has_duplicates = any(normalized.count(norm_p) == 2 for norm_p in set(normalized) if norm_p)
        prefixed_dublet = any(pid.endswith("_") for pid in params) and any(pid.startswith("_") for pid in params)
        params_of_different_fields = any(p not in modules_params for p in params) or any(p not in modules_return_map for p in params)
        if has_duplicates or prefixed_dublet or params_of_different_fields:
            return True
        return False

    def classify_equations_for_module(
        self,
        methods: list,
    ) -> dict:
        """
        Classify equations for a specific MODULE into a dict with keys:
        - differential: method includes same param min 2 times (param.replace("_","") for param in params)
        - interaction: method params originate from min 2 fields of different types
        - cor: method requires params of just single field type, or uses return_key of other methods in the module
        """
        try:
            classification = {
                "differential": [],
                "cor": []
            }

            for mid, mattrs in methods:
                if (mattrs or {}).get("type") != "METHOD":
                    continue
                params = mattrs.get("params")
                code = mattrs.get("code")
                if not params or not isinstance(params, list):
                    continue

                if any(p.startswith("_") or p.endswith("_") for p in params):
                    classification["differential"].append(code)
                else:
                    classification["cor"].append(code)
            print("classification completed...")
            return classification
        except Exception as e:
            print("Err classify_equations_for_module failed", e)
            return {"differential": [], "cor": []}

    def method_layer(self, modules):
        # todo classify form all modules into single eq classification struct -> build ctlr
        # For each method: use params and neighbor_vals to collect the params index for each item of
        # neighbor_vals within a list (if neighbor_vals else None), and append it to a NEIGHBOR_CTLR
        # struct under the same index as the specific method (and overlying module_idx).
        """
        Method layer for the Guard workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `modules`.
        2. Builds `method_struct` and writes current `self.sim_time`, `self.amount_nodes`, `self.dims` onto each graph METHOD node (method tree).
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `print()`, `len()`, `range()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `modules`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        print("method_layer... ")
        mod_len_exclude_ghost = len(modules) -1

        method_struct = {
            "METHOD_PARAM_LEN_CTLR": [[] for _ in range(mod_len_exclude_ghost)],
            "METHODS": [[] for _ in range(mod_len_exclude_ghost)],
            "METHODS_PER_MOD_LEN_CTLR": [0 for _ in range(mod_len_exclude_ghost)],
        }

        mnames = [[] for _ in range(mod_len_exclude_ghost)]
        try:
            for mid, module in modules:
                # ghost does not have equation
                if "GHOST" in mid.upper(): continue
                print("method_layer work", mid)

                midx:int = module.get("module_index")

                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                )

                mids = [defid for defid, defattrs in methods]
                mlen: int = len(mids)

                if not mlen:
                    print("method_layer... len methods", mlen)
                    continue

                # Classification struct per module
                method_struct["METHODS"][midx] = [muattrs.get("code") for muid, muattrs in methods]

                print(f"method_struct[METHODS][midx] for {mid} set")

                # LEN PARAMS / METHOD
                for mu_id, mu_attrs  in methods:
                    params_len = len(mu_attrs["params"])
                    method_struct[
                        "METHOD_PARAM_LEN_CTLR"
                    ][midx].append(params_len)
                    # CHAR: same run spec on every METHOD node for inspectors / QF / downstream
                    if self.g is not None and mu_id is not None:
                        self.g.update_node({
                            "id": mu_id,
                            "sim_time": int(self.sim_time),
                            "amount_nodes": int(self.amount_nodes),
                            "dims": int(self.dims),
                        })

                # len methods per module
                method_struct["METHODS_PER_MOD_LEN_CTLR"][midx] = mlen
                print(f"METHODS_PER_MOD_LEN_CTLR for {mid} set")

                mnames[midx].extend(mids)
                print(f"mnames for {mid} set")

            flatten_ctlr = []
            for sublist in method_struct["METHOD_PARAM_LEN_CTLR"]:
                flatten_ctlr.extend(sublist)
            method_struct["METHOD_PARAM_LEN_CTLR"] = flatten_ctlr
            print(f"METHOD_PARAM_LEN_CTLR finished", method_struct["METHOD_PARAM_LEN_CTLR"])

            # flatten mnames
            flatten_mnames = []
            for item in mnames:
                flatten_mnames.extend(item)

            # FLATTEN METHODS
            flatten_methods = []
            for m in method_struct["METHODS"]:
                flatten_methods.extend(m)
            method_struct["METHODS"] = flatten_methods

            print("=== METHOD ID -> IDX CTLR ========")
            for i, defid in enumerate(flatten_mnames):
                print(f"{i} - {defid}")
            print("==================================")

        except Exception as e:
            print("Err method_layer", e)
        print(f"method_layer... done")
        return method_struct


    def set_eq_operator_ctlr(self, modules):
        """
        Set eq operator ctlr for the Guard workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `modules`.
        2. Builds intermediate state such as `midx`, `methods`, `mids` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `module.get()`, `self.g.get_neighbor_list_rel()`, `list()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `modules`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        for mid, module in modules:
            # ghost does not have equation
            if "GHOST" in mid.upper(): continue
            # print("method_layer... working", mid)

            midx: int = module.get("module_index")

            methods = self.g.get_neighbor_list_rel(
                trgt_rel="has_method",
                node=mid,
                as_dict=True,
            )

            mids = list(methods.keys())
            mlen: int = len(mids)

            if not mlen:
                print("method_layer... len methods", mlen)
                continue

            # Iterate Equations
            for eqid, eqattrs in methods.items():
                params = eqattrs.get("params")
                equation = eqattrs.get("equation")
                print("eq extractted", equation)

                # set operator map eq based
                self.operator_handler.process_code(
                    code=equation,
                    params=params,
                    midc=midx
                )
        return self.operator_handler.operator_pathway_ctlr




    def set_edge_method_to_db(self):
        """
        Map method return_key indices to DB slots per module.
        Each method's return_key must exist in a field's keys; only then is the
        (module, field, param_index) added. Parses keys from JSON if stored as string.
        """
        # each eqs fields has different return key (IMPORTANT: sum variation results)
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE"
        )

        return_key_map = [
            []
            for _ in range(len(modules)-1)
        ]

        try:
            import json
            print("start compilation...")
            for mid, module in modules:
                if "GHOST" in mid.upper(): continue
                print("set_edge_method_to_db... working", mid)

                m_idx = module.get("module_index")
                print("m_idx", m_idx)
                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                if not len(list(methods.keys())):
                    print("set_edge_method_to_db... len methods 0")
                    continue

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                print("fields", mid, len(fields))
                for eqid, eqattrs in methods.items():
                    if (eqattrs or {}).get("type") != "METHOD":
                        continue
                    return_key = eqattrs.get("return_key")
                    if not return_key:
                        continue

                    for fidx, (fid, fattrs) in enumerate(fields.items()):
                        if (fattrs or {}).get("type") != "FIELD":
                            continue
                        keys = fattrs.get("keys", [])
                        if isinstance(keys, str):
                            keys = json.loads(keys) if keys else []

                        if return_key not in keys:
                            continue

                        field_index = fattrs["field_index"]
                        rindex = keys.index(return_key)

                        return_key_map[m_idx].append(
                            self.get_db_index(
                                m_idx,
                                field_index,
                                rindex,
                            )
                        )
            # flatte
            flatten_rtk_map = []
            for module_items in return_key_map:
                flatten_rtk_map.extend(module_items)
            return flatten_rtk_map
        except Exception as e:
            print("Err set_edge_method_to_db", e)
            # Return valid empty structure to avoid None breaking grid
            return []


    def set_edge_db_to_method(self, modules, env_id):
        """
        Set edge db to method for the Guard workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `modules`, `env_id`.
        2. Builds intermediate state such as `mlen`, `db_to_method`, `ghost_fields` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `print()`, `self.g.get_neighbor_list_rel()`, `len()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `modules`: Caller-supplied value used during processing.
        - `env_id`: Identifier used to target the relevant entity.

        Returns:
        - Returns the computed result for the caller.
        """
        print("set_edge_db_to_method...")
        # todo for field index includ again
        mlen = len(modules)-1


        # db out gnn
        db_to_method = {
            "DB_TO_METHOD_EDGES": [[] for _ in range(mlen)],
            "DB_CTL_VARIATION_LEN_PER_EQUATION": [[] for _ in range(mlen)],
            "DB_CTL_VARIATION_LEN_PER_FIELD": [[] for _ in range(mlen)],
            "LEN_FEATURES_PER_EQ": self.get_empty_method_structure(set_zero=False),
        }

        # CHAR: LEN_FEATURES_PER_EQ is flattened by module_index order, so build
        # a stable global method offset per module before filling it.
        method_counts_by_module_index = [0 for _ in range(mlen)]
        for mid, module in modules:
            if "GHOST" in mid.upper():
                continue
            module_index = module.get("module_index")
            if module_index is None or module_index >= mlen:
                continue

            method_nodes = self.g.get_neighbor_list_rel(
                trgt_rel="has_method",
                node=mid,
                as_dict=True,
            )
            method_counts_by_module_index[module_index] = len(method_nodes)

        method_offsets_by_module_index = {}
        running_method_offset = 0
        for module_index, method_count in enumerate(method_counts_by_module_index):
            method_offsets_by_module_index[module_index] = running_method_offset
            running_method_offset += method_count

        ghost_fields = self.g.get_neighbor_list_rel(
            trgt_rel="has_field",
            node="GHOST_MODULE",
        )

        try:
            print("start compilation...")
            for mid, module in modules:
                if "GHOST" in mid.upper(): continue
                m_idx = module.get("module_index")

                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )
                if _cor_verbose():
                    print(f"mid {mid} has methods {len(methods)}")
                if not methods:
                    print("set_edge_db_to_method... len methods 0")
                    continue

                if m_idx is None:
                    continue

                fields = self.g.get_neighbor_list_rel(
                    node=mid, trgt_rel="has_field"
                )

                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    try:
                        # CHAR: MultiGraph returns all neighbors — skip non-METHOD nodes
                        if (eqattrs or {}).get("type") != "METHOD":
                            continue
                        global_eq_idx = method_offsets_by_module_index.get(m_idx, 0) + eq_idx
                        EQ_AMOUNT_VRIATIONS_FIELDS = 0

                        # VALIDATE PARAM
                        params = eqattrs.get("params", [])
                        if isinstance(params, str):
                            params = json.loads(params) if params else []
                        if params is None or not isinstance(params, list):
                            print("params unsupported format", type(params))
                            params = []

                        params_origin = eqattrs.get("origin", [])
                        if isinstance(params_origin, str):
                            params_origin = json.loads(params_origin) if params_origin else []
                        if not params_origin:
                            params_origin = [""] * len(params)

                        for fidx, (fid, fattrs) in enumerate(fields):
                            field_index = fattrs["field_index"]
                            field_eq_param_struct = []

                            keys = fattrs.get("keys", [])
                            if isinstance(keys, str):
                                keys = json.loads(keys)
                            fattrs["keys"] = keys

                            fneighbors = self.g.get_neighbor_list_rel(
                                node=fid,
                                trgt_rel="has_interactant",
                            )

                            # todo if 3 (or more) identical keys include dict field, bool - map
                            worked_params = {}
                            for pidx, pid in enumerate(params):
                                pid_orig = pid  # keep original for worked_params lookup
                                if pid not in worked_params:
                                    # space collect fields
                                    worked_params[pid] = []

                                collected = False

                                param_collector = []
                                param_origin_key_collector = []
                                if _cor_verbose():
                                    print("work pid", pid)

                                # Field's own param
                                is_prev_pre = pid.startswith("prev_")
                                is_prev_after = pid.endswith("_prev")
                                is_self_prefixed = pid.startswith("_")

                                # rm _ before and after
                                filtered_key = clean_underscores_front_back(
                                    text=pid
                                )

                                final_key = rm_prev_mark(filtered_key)

                                # PRE CHECK METHOD PARAM ORIGINS
                                is_self_param:bool = self.self_param(
                                    current_param=final_key,
                                    self_field_params=keys,
                                )

                                # SELF PARAM?
                                if is_self_param is True:
                                    # Add field to param collection
                                    worked_params[pid_orig].append(fid)

                                    time_dim = self.get_time_dim(
                                        is_prev_pre, is_prev_after, is_self_prefixed,
                                    )

                                    pindex = keys.index(final_key)

                                    result = self.get_db_index(
                                        m_idx,
                                        field_index,
                                        pindex,
                                        time_dim,
                                    )

                                    field_eq_param_struct.append(
                                        result
                                    )

                                    collected = True
                                    continue
                                else:
                                    # COLLECT PARAM FROM NEIGHBOR — skip non-FIELD nodes
                                    for finid, fiattrs in fneighbors:
                                        if (fiattrs or {}).get("type") != "FIELD":
                                            continue
                                        ikeys = fiattrs.get("keys")

                                        if isinstance(ikeys, str):
                                            ikeys = json.loads(ikeys)
                                        if not ikeys:
                                            continue

                                        # param key in interactant field?
                                        if final_key in ikeys and finid not in worked_params[pid_orig]:

                                            # Add field to param collection
                                            worked_params[pid_orig].append(fid)

                                            fmod = self.g.get_node(fiattrs.get("module_id"))

                                            nfield_index = fiattrs["field_index"]

                                            pindex = ikeys.index(final_key)

                                            param_collector.append(
                                                self.get_db_index(
                                                    (fmod or {}).get("module_index", 0),
                                                    nfield_index,
                                                    pindex,
                                                )
                                            )
                                            # collect field interaction keys
                                            param_origin_key_collector.append(finid)
                                            collected = True

                                    for gfi, (gfid, gfattrs) in enumerate(ghost_fields):
                                        if (gfattrs or {}).get("type") != "FIELD":
                                            continue
                                        gikeys = gfattrs.get("keys")

                                        if isinstance(gikeys, str):
                                            gikeys = json.loads(gikeys)
                                        if not gikeys:
                                            continue

                                        if final_key in gikeys and gfid not in worked_params[pid_orig]:

                                            # Add field to param collection
                                            worked_params[pid_orig].append(fid)

                                            gfield_index = gfattrs["field_index"]

                                            pindex = gikeys.index(final_key)
                                            #print("interactant pindex", pindex)
                                            #print(f"{pid} found in gfid ({gfield_index}) (pindex {pindex})", gfield_index)

                                            gmod = self.g.get_node("GHOST_MODULE")

                                            # ADD PARAM TO
                                            param_collector.append(
                                                self.get_db_index(
                                                    (gmod or {}).get("module_index", 0),
                                                    gfield_index,
                                                    pindex,
                                                )
                                            )
                                            collected = True

                                    if collected is False:
                                        # Fallback: param not in field/interactant/ghost keys
                                        gmod = self.g.get_node("GHOST_MODULE")
                                        env_field = self.g.get_node(env_id) or {}
                                        env_field_index = env_field.get("field_index", 0)
                                        ekeys = env_field.get("keys") or []
                                        if isinstance(ekeys, str):
                                            ekeys = json.loads(ekeys) if ekeys else []
                                        pindex = ekeys.index("None") if "None" in ekeys else 0
                                        param_collector.append(
                                            self.get_db_index(
                                                gmod.get("module_index", 0),
                                                env_field_index,
                                                pindex,
                                            )
                                        )
                                if _cor_verbose():
                                    print(f"add {len(param_collector)} to field_eq_param_struct")
                                field_eq_param_struct.append(param_collector)

                            # Upscale variation struct
                            expand_field_eq_variation_struct = expand_structure(
                                struct=field_eq_param_struct
                            )
                            if _cor_verbose():
                                print(f"expand_field_eq_variation_struct for {eqid}", len(expand_field_eq_variation_struct), expand_field_eq_variation_struct)
                            param_struct = {}
                            for key, value in zip(params, expand_field_eq_variation_struct):
                                param_struct[key] = value

                            #db_to_method["VARIATION_INDICES"][m_idx].append(param_struct)

                            # extend variation single eq
                            for item in expand_field_eq_variation_struct:
                                db_to_method["DB_TO_METHOD_EDGES"][m_idx].extend(item)
                                #print(f"module {mid} ({m_idx}) expand_field_eq_variation_struct item {eqid}", item)
                                # todo calc just / len(method_param) to sort them

                            field_variations_eq = len(expand_field_eq_variation_struct)
                            EQ_AMOUNT_VRIATIONS_FIELDS += field_variations_eq

                            # add len field var to emthod struct
                            if global_eq_idx < len(db_to_method["LEN_FEATURES_PER_EQ"]):
                                db_to_method[
                                    "LEN_FEATURES_PER_EQ"
                                ][global_eq_idx].append(field_variations_eq)

                            db_to_method[
                                "DB_CTL_VARIATION_LEN_PER_FIELD"
                            ][m_idx].append(len(expand_field_eq_variation_struct))

                        db_to_method[
                            "DB_CTL_VARIATION_LEN_PER_EQUATION"
                        ][m_idx].append(EQ_AMOUNT_VRIATIONS_FIELDS)
                    except Exception as e:
                        print("Err set_edge_db_to_method", e, eqid)

            flatten_variations = []
            for i, item in enumerate(db_to_method["DB_TO_METHOD_EDGES"]):
                flatten_variations.extend(item)
            db_to_method["DB_TO_METHOD_EDGES"] = flatten_variations

            flatten_amount_variations = []
            for item in db_to_method["DB_CTL_VARIATION_LEN_PER_EQUATION"]:
                flatten_amount_variations.extend(item)
            db_to_method["DB_CTL_VARIATION_LEN_PER_EQUATION"] = flatten_amount_variations

            # for sepparation for the sum process
            flatten_amount_variations_per_field = []
            for item in db_to_method["DB_CTL_VARIATION_LEN_PER_FIELD"]:
                flatten_amount_variations_per_field.extend(item)
            db_to_method["DB_CTL_VARIATION_LEN_PER_FIELD"] = flatten_amount_variations_per_field

            print("set_edge_db_to_method... done")
        except Exception as e:
            print(f"Err set_edge_db_to_method: {e}")
            raise
        return db_to_method

    def self_param(self, self_field_params, current_param):
        external_param:bool = current_param.endswith("_")
        in_self_params:bool = current_param in self_field_params
        is_self_prefixed = current_param.startswith("_")
        if external_param is True:
            if _cor_verbose():
                print(f"{current_param} in xternal field")
            return False
        elif in_self_params is True or is_self_prefixed:
            if _cor_verbose():
                print(f"{current_param} in self field")
            return True



    def get_time_dim(self, is_prev_pre, is_prev_after, is_self_prefixed):
        time_dim = None

        if is_prev_pre:
            time_dim = 1
        elif is_prev_after:
            time_dim = 1

        # directly sort in param arrays
        if time_dim is None:
            time_dim = 0
        return time_dim


    def set_iterator_from_humans(self):
        """
        Set iterator from humans for the Guard workflow.

        Workflow:
        1. Starts from the current object state and local workflow context.
        2. Builds intermediate state such as `iterator`, `modules`, `ghost_fields` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `self.g.get_nodes()`, `self.g.get_neighbor_list_rel()`, `enumerate()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - None.

        Returns:
        - Returns the computed result for the caller.
        """
        iterator = {
            "MODULES": [],
            "FIELDS": [],
        }
        import json
        modules = self.g.get_nodes(filter_key="type", filter_value="MODULE")

        ghost_fields = self.g.get_neighbor_list(
            target_type="FIELD", node="GHOST_MODULE",
        )

        try:
            for i, (mid, module) in enumerate(modules):
                mod_idx = module.get("module_index", i)
                # A. FeDB_PARAM_CONTROLLERlder des Moduls sammeln
                fields = self.g.get_neighbor_list_rel(
                    node=mid, trgt_rel="has_field"
                )
                print("field ids", [f[0] for f in fields])
                len_fields = len(fields)
                iterator["FIELDS"].append(len_fields)

                # B. Methoden (Gleichungen) des Moduls sammeln
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method", node=mid, as_dict=True
                )
                iterator["MODULES"].append(len(methods))

                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    if (eqattrs or {}).get("type") != "METHOD":
                        continue
                    params = eqattrs.get("params", [])
                    if isinstance(params, str):
                        params = json.loads(params) if params else []
                    if params is None or not isinstance(params, list):
                        params = []
                    params_origin = eqattrs.get("origin", None) or ["" for _ in range(len(params))]

                    for fidx, (fid, fattrs) in enumerate(fields):
                        # Vorbereitung der Keys des aktuellen Feldes
                        keys = fattrs.get("keys", [])
                        if isinstance(keys, str): keys = json.loads(keys)
                        print("fattrs fid", fid)
                        field_index = fattrs["field_index"]

                        fneighbors = self.g.get_neighbor_list_rel(
                            node=fid,
                            trgt_rel="has_interactant",
                            as_dict=True,
                        )

                        # Struktur für die Parameter-Zuweisung dieses Feldes
                        field_eq_param_struct = []

                        for pidx, pid in enumerate(params):
                            is_prefixed = pid.endswith("_")
                            clean_pid = pid[:-1] if is_prefixed else pid
                            collected = False
                            param_collector = []

                            # 1. Check: Gehört der Parameter zum Feld selbst?
                            if clean_pid in keys and not is_prefixed and params_origin[pidx] not in ["neighbor", "interactant"]:
                                pindex = keys.index(clean_pid)

                                field_eq_param_struct.append(
                                    self.get_db_index(
                                        mod_idx,
                                        field_index,
                                        pindex,
                                    )
                                )
                                collected = True
                            else:
                                # 2. Check: Ist es ein Interactant (Neighbor)?
                                for finid, fiattrs in fneighbors.items():
                                    if (fiattrs or {}).get("type") != "FIELD":
                                        continue
                                    ikeys = fiattrs.get("keys", [])
                                    if isinstance(ikeys, str): ikeys = json.loads(ikeys)
                                    if not ikeys: continue

                                    if clean_pid in ikeys:
                                        pindex = ikeys.index(clean_pid)
                                        nfield_index = fiattrs["field_index"]
                                        pmod_id = fiattrs.get("module_id")
                                        if not pmod_id or not self.g.G.has_node(pmod_id):
                                            continue
                                        pmod = self.g.get_node(pmod_id)
                                        param_collector.append(
                                            self.get_db_index(
                                                pmod.get("module_index", 0),
                                                nfield_index,
                                                pindex,
                                            ))
                                        collected = True

                                # 3. Check: Ghost-Felder (Globaler Fallback)
                                if not collected:
                                    for gfi, (gfid, gfattrs) in enumerate(ghost_fields):
                                        if (gfattrs or {}).get("type") != "FIELD":
                                            continue
                                        gikeys = gfattrs.get("keys", [])
                                        gfield_index = gfattrs["field_index"]
                                        pmod = self.g.get_node("GHOST_MODULE")

                                        if isinstance(gikeys, str): gikeys = json.loads(gikeys)
                                        if clean_pid in gikeys:
                                            pindex = gikeys.index(clean_pid)
                                            param_collector.append(
                                                self.get_db_index(
                                                    pmod.get("module_index", 0),
                                                    gfield_index,
                                                    pindex,
                                                )
                                            )
                                            collected = True

                                field_eq_param_struct.append(param_collector if collected else -1)

            #print("set_iterator_from_humans: GPU Skeleton successfully compiled.")
        except Exception as e:
            print(f"Error in set_iterator_from_humans: {e}")
            raise e
        return iterator



    def sync_field_keys_from_methods(self, dims: int = None):
        print("sync_field_keys_from_methods...")
        try:
            # Resolve dims from ENV node or default
            if dims is None:
                env_nodes = self.g.get_nodes(filter_key="type", filter_value="ENV")
                dims = 3
                for _nid, env_attrs in env_nodes:
                    dims = env_attrs.get("dims", 3)
                    break

            modules = self.g.get_nodes(filter_key="type", filter_value="MODULE")
            for mid, _module in modules:
                if "GHOST" in mid.upper():
                    continue

                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )
                if not methods:
                    continue

                fields = self.g.get_neighbor_list_rel(
                    trgt_rel="has_field",
                    node=mid,
                    as_dict=True,
                )
                if not fields:
                    continue

                # Collect all required keys from methods (params + return_key)
                required_keys = set()
                for _eqid, eqattrs in methods.items():
                    params = eqattrs.get("params", [])
                    if isinstance(params, str):
                        try:
                            params = json.loads(params) if params else []
                        except json.JSONDecodeError:
                            params = []
                    if params is None or not isinstance(params, list):
                        params = []
                    return_key = eqattrs.get("return_key")
                    if return_key:
                        required_keys.add(return_key)
                    for p in params:
                        if p:
                            required_keys.add(p)

                # For each field, add missing keys with default value and axis_def 0
                for fid, fattrs in fields.items():
                    keys = fattrs.get("keys", [])
                    if isinstance(keys, str):
                        try:
                            keys = json.loads(keys) if keys else []
                        except json.JSONDecodeError:
                            keys = []
                            
                    values = fattrs.get("values", [])
                    if isinstance(values, str):
                        try:
                            values = json.loads(values) if values else []
                        except json.JSONDecodeError:
                            values = []

                    axis_def = fattrs.get("axis_def", [])
                    if isinstance(axis_def, str):
                        try:
                            axis_def = json.loads(axis_def) if axis_def else []
                        except json.JSONDecodeError:
                            axis_def = []
                    # Pad axis_def to match keys length if needed
                    while len(axis_def) < len(keys):
                        axis_def.append(None)

                    missing_keys = [k for k in required_keys if k not in keys]
                    if not missing_keys:
                        continue

                    default_val = [0 for _ in range(dims)]
                    for key in missing_keys:
                        keys.append(key)
                        values.append(default_val)
                        axis_def.append(0)

                    self.g.update_node({
                        "id": fid,
                        "keys": keys,
                        "values": values,
                        "axis_def": axis_def,
                    })
        except Exception as e:
            print("Err sync_field_keys_from_methods", e)
        print("sync_field_keys_from_methods... done")

    def get_modules_methods(self, mid):

        methods = {}
        method_nodes = self.g.get_neighbor_list_rel(
            trgt_rel="has_method",
            node=mid,
            as_dict=True,
        )
        methods.update(method_nodes)

        params = self.g.get_neighbor_list_rel(
            trgt_rel="has_param",
            node=mid,
            as_dict=True,
        )

        for pid, pattrs in params.items():
            if "code" in pattrs:
                # param
                methods[pid] = pattrs


    def create_actor(self):

        try:
            import ray
            @ray.remote
            class GuardWorker(Guard):
                def __init__(
                        self,
                        qfu,
                        g,
                        user_id
                ):
                    Guard.__init__(
                        self,
                        qfu,
                        g,
                        user_id
                    )
        except Exception as e:
            print("Ray not accessible:", e)


    def get_module_eqs(self, mid):
        # get methods for module
        meq = self.g.get_neighbor_list(
            node=mid,
            target_type="METHOD",
        )

        # bring to exec order
        meq = sorted(meq.values(), key=lambda x: x["method_index"], reverse=True)
        return meq


    def get_empty_field_structure(self, include_ghost_mod=True):
        modules: list = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )
        len_mods = len(modules) if include_ghost_mod is True else len(modules) - 1
        modules_struct = [[] for _ in range(len_mods)]
        try:
            print("modules_struct initialized size:", len(modules))

            for mid, m in modules:
                if include_ghost_mod is False:
                    if "GHOST" in mid.upper(): continue
                midx = m["module_index"]

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field"
                )

                keys = [item[0] for item in fields]

                print(f"{mid} fields: {keys}")
                field_struct = []
                for _ in range(len(keys)):
                    field_struct.append([])

                # SET EMPTY FIELDS STRUCT AT MODULE INDEX
                modules_struct[midx]=field_struct
        except Exception as e:
            print("Err get_empty_field struct:", e)
        return modules_struct

    def get_db_index(self, mod_idx, field_idx, param_in_field_idx, time_dim=0):
        return (
            time_dim, # the t-dim to choose the item from
            mod_idx,
            field_idx,
            param_in_field_idx,
        )

    def get_empty_method_structure(self, set_zero=True, ):

        modules: list = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        mlen_excl_ghost = len(modules)-1

        method_struct = [
            [] if set_zero is False else 0
            for _ in range(mlen_excl_ghost)
        ]

        try:
            # nit with size max_index + 1
            modules_struct = []
            print("modules_struct initialized size:", len(modules_struct))

            for mid, m in modules:
                if "GHOST" in mid.upper(): continue

                module_index = m.get("module_index")
                if module_index is None:
                    continue

                method_nodes = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                method_struct[module_index] = [
                    [] if set_zero is False else 0
                    for _ in range(len(method_nodes))
                ]

            flattened_methods = []
            for i in method_struct:
                flattened_methods.extend(i)

            return flattened_methods
        except Exception as e:
            print("Err get_empty_field struct:", e)
