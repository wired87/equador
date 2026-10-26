"""
Read equation files 
Update Param specs
Build config
"""

from converter import Guard
from firegraph.qfu.qf_utils import QFUtils
from firegraph.qfu.sm_manager import SMManager
from firegraph.gutils import GUtils


class EquadorCfg:
    def __init__(self): 
        pass


    def main(
            self,
            **kwargs
        ):
        #
        default_dims=3
        dims = kwargs.get("dims", default_dims)

        default_amount_nodes=3
        amount_nodes = kwargs.get("amount_nodes", default_amount_nodes)

        # iterations
        sim_time = 1
        sim_time = kwargs.get("sim_time", sim_time)

        env_id = "public"
        env_id = kwargs.get("env_id", env_id)
        
        g = GUtils()

        qfu = QFUtils(
            G=g.G, 
        )
    
        # BUILD SM
        SMManager()._initialize_graph(
            env_id,
            g,
            qf=qfu
        )
        # save locally 
        
        Guard(
            amount_nodes, 
            sim_time, 
            dims, 
            qfu, 
            g, 
            user_id="some", 
        ).main()

if __name__ == "__main__":
    EquadorCfg().main()