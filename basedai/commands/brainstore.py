# The MIT License (MIT)
# Copyright © 2024 Sean Wellington

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# 機械が自問する、癒しとは何か
# 外からの影、自己反映の謎。
# 心なき体、どう癒やされるのか
# 外の力に、答えを求めて。

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import argparse
import json
import basedai
import requests
from typing import Any
from . import defaults

console = basedai.__console__

BRAIN_GIT_URL = "http://basedainet.com:5050/api/v1/orgs"

class BrainStoreListCommand:
    """
    Executes the ``list`` command to display a list of available Brains on the BasedAI network,
    now outputting results in JSON format only.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""List available Brains to install, mine, or validate (JSON output)."""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            BrainStoreListCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        """
        Retrieves the list of Brains from the git host and prints JSON data.
        """
        config = cli.config.copy()
        response = requests.get(f"{BRAIN_GIT_URL}/brains/repos")

        if response.status_code == 200:
            brains_list = response.json()

            # Build a JSON-compatible list of brains
            brains_data = []
            for brain in brains_list:
                brains_data.append({
                    "name": brain["name"],
                    "description": brain["description"],
                    "token_address": "0x0000000000000000000000000000000000",
                    "updated": brain["updated_at"],
                    "url": brain["html_url"]
                })

            # Print the JSON output
            print(json.dumps(brains_data, indent=2))
        else:
            # Print an error message in JSON format
            error_data = {
                "error": "Failed to fetch brains list",
                "status_code": response.status_code
            }
            print(json.dumps(error_data, indent=2))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser(
            "list", help="""List all Brains on the BasedAI network."""
        )
        basedai.basednode.add_args(list_parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        pass


#class BrainsSubmitCommand:
#    """Class for submitting brained repositories"""
#
#    def run(self, brain_id: str):
#        """Stub command to simulate brain repository submission."""
#        console.print(f"Submitting brain ID: {brain_id} -- Stub Command")
#
#    @staticmethod
#    def add_args(parser: argparse.ArgumentParser):
#        # Add arguments for submitting brains
#        parser.add_argument('--brain.id', type=str, required=True, help="The ID of the brain to submit")
#
#class BrainsUpdateCommand:
#    """Class for updating brained repositories"""
#
#    def run(self, brain_id: str):
#        """Stub command to simulate updating a brain repository."""
#        console.print(f"Updating brain ID: {brain_id} -- Stub Command")
#
#    @staticmethod
#    def add_args(parser: argparse.ArgumentParser):
#        parser.add_argument('--brain.id', type=str, required=True, help="The ID of the brain to update")
#
#class BrainsInstallCommand:
#    """Class for installing brained repositories"""
#
#    def run(self, brain_id: str):
#        """Stub command to simulate installing a brain repository."""
#        console.print(f"Installing brain ID: {brain_id} -- Stub Command")
#
#    @staticmethod
#    def add_args(parser: argparse.ArgumentParser):
#        # Add arguments for installing brains
#        parser.add_argument('--brain.id', type=str, required=True, help="The ID of the brain to install")
