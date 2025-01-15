import time
import argparse
import basedai
import hashlib
from . import defaults
from substrateinterface.utils.ss58 import ss58_decode
from typing import List, Optional, Dict
from .utils import get_delegates_details, DelegatesDetails, check_netuid_set
import json  # <-- Import JSON for output

console = basedai.__console__

def ss58_to_ethereum(ss58_address):
    return ss58_address

class LinkBrainCommand:
    @staticmethod
    def run(cli: "basedai.cli"):
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            LinkBrainCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        wallet = basedai.wallet(config=cli.config)
        basednode.register_subnetwork(
            wallet=wallet,
            prompt=not cli.config.no_prompt,
        )

    @classmethod
    def check_config(cls, config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "link",
            help="""Open a link to associate a Brain to a wallet.""",
        )
        basedai.wallet.add_args(parser)
        basedai.basednode.add_args(parser)

class BrainListCommand:
    """
    The "list" command is executed to enumerate all active Brains along with their detailed network information,
    but now outputs JSON instead of a Rich table.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            BrainListCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        """List all Brains on the network in JSON."""
        subnets: List[basedai.BrainInfo] = basednode.get_all_brains_info()

        delegate_info: Optional[Dict[str, DelegatesDetails]] = get_delegates_details(
            url=basedai.__delegates_details_url__
        )

        all_brains = []
        total_neurons = 0

        for subnet in subnets:
            total_neurons += subnet.max_n
            all_brains.append({
                "BRAIN ID": str(subnet.netuid),
                "EMISSION": f"{subnet.emission_value / basedai.utils.RAOPERBASED * 100:0.2f}%",
                "TEMPO": str(subnet.tempo),
                "WORK": str(basedai.utils.formatting.millify(subnet.difficulty)),
                "AGENTS": str(subnet.subnetwork_n),
                "AGENT LIMIT": str(basedai.utils.formatting.millify(subnet.max_n)),
                "AGENT FEE": f"{subnet.burn!s:8.8}",
                "SMART CONTRACTS": "0",
                "BASED ADDRESS": (
                    delegate_info[subnet.owner_ss58].name
                    if subnet.owner_ss58 in delegate_info
                    else subnet.owner_ss58
                ),
            })

        # You can also include summary info in JSON if desired
        output_data = {
            "total_brains": len(subnets),
            "total_agents": total_neurons,
            "brains": all_brains
        }

        # Print JSON to stdout
        print(json.dumps(output_data, indent=2))

    @staticmethod
    def check_config(config: "basedai.config"):
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_subnets_parser = parser.add_parser(
            "list", help="""List all Brains on the network (JSON output)"""
        )
        basedai.basednode.add_args(list_subnets_parser)


HYPERPARAMS = {
    "serving_rate_limit": "sudo_set_serving_rate_limit",
    "min_difficulty": "sudo_set_min_difficulty",
    "max_difficulty": "sudo_set_max_difficulty",
    "weights_version": "sudo_set_weights_version_key",
    "weights_rate_limit": "sudo_set_weights_set_rate_limit",
    "max_weight_limit": "sudo_set_max_weight_limit",
    "immunity_period": "sudo_set_immunity_period",
    "min_allowed_weights": "sudo_set_min_allowed_weights",
    "activity_cutoff": "sudo_set_activity_cutoff",
    "network_registration_allowed": "sudo_set_network_registration_allowed",
    "network_pow_registration_allowed": "sudo_set_network_pow_registration_allowed",
    "min_burn": "sudo_set_min_burn",
    "max_burn": "sudo_set_max_burn",
}

class BrainSetParametersCommand:
    @staticmethod
    def run(cli: "basedai.cli"):
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            BrainSetParametersCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(
        cli: "basedai.cli",
        basednode: "basedai.basednode",
    ):
        wallet = basedai.wallet(config=cli.config)
        print("\n")
        BrainParametersCommand.run(cli)
        if not cli.config.is_set("param") and not cli.config.no_prompt:
            param = Prompt.ask("Enter parameter", choices=HYPERPARAMS)
            cli.config.param = str(param)
        if not cli.config.is_set("value") and not cli.config.no_prompt:
            value = Prompt.ask("Enter new value")
            cli.config.value = value

        if (
            cli.config.param == "network_registration_allowed"
            or cli.config.param == "network_pow_registration_allowed"
        ):
            cli.config.value = True if cli.config.value.lower() == "true" else False

        basednode.set_hyperparameter(
            wallet,
            netuid=cli.config.netuid,
            parameter=cli.config.param,
            value=cli.config.value,
            prompt=not cli.config.no_prompt,
        )

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(
                config, basedai.basednode(config=config, log_verbose=False)
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("set", help="""Set parameters for a Brain""")
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        parser.add_argument("--param", dest="param", type=str, required=False)
        parser.add_argument("--value", dest="value", type=str, required=False)

        basedai.wallet.add_args(parser)
        basedai.basednode.add_args(parser)

class BrainParametersCommand:
    @staticmethod
    def run(cli: "basedai.cli"):
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            BrainParametersCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        subnet: basedai.BrainHyperparameters = basednode.get_subnet_hyperparameters(
            cli.config.netuid
        )
        table = basedai.__console__.table(
            title=f"[white]BRAIN PARAMETERS - ID: {cli.config.netuid} - {basednode.network}"
        )
        table.add_column("[overline white]PARAMETER", style="bold white")
        table.add_column("[overline white]VALUE", style="cyan")

        for param in subnet.__dict__:
            table.add_row("  " + param, str(subnet.__dict__[param]))

        basedai.__console__.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(
                config, basedai.basednode(config=config, log_verbose=False)
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "parameters", help="""View Brain Parameters"""
        )
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        basedai.basednode.add_args(parser)

class BrainGetParametersCommand:
    @staticmethod
    def run(cli: "basedai.cli"):
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            BrainGetParametersCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        subnet: basedai.BrainHyperparameters = basednode.get_subnet_hyperparameters(
            cli.config.netuid
        )
        table = basedai.__console__.table(
            title=f"[white]Brain Rules - ID: {cli.config.netuid} - {basednode.network}"
        )
        table.add_column("[overline white]PARAMETER", style="white")
        table.add_column("[overline white]VALUE", style="cyan")

        for param in subnet.__dict__:
            table.add_row(param, str(subnet.__dict__[param]))

        basedai.__console__.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(
                config, basedai.basednode(config=config, log_verbose=False)
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("get", help="""View Brain parameters""")
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        basedai.basednode.add_args(parser)
