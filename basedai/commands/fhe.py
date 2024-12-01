import argparse
import basedai
from basedai.commands import BasedCommand
import json
import asyncio
import websockets
import os
import tenseal as ts
import concrete.numpy as cnp
import pyfhel
from phe import paillier
import ollama
from typing import Any, Dict
import logging
import secrets
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FHEError(Exception):
    """Base exception for FHE-related errors."""
    pass

class FHERunCommand(BasedCommand):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--address', type=str, required=True, help='Address that signed the work')
        parser.add_argument('--balance', type=float, required=True, help='Minimum balance required')
        parser.add_argument('--command', type=str, required=True, help='FHE command to run')
        parser.add_argument('--library', type=str, choices=['tenseal', 'concrete', 'pyfhel', 'paillier'], required=True, help='FHE library to use')
        parser.add_argument('--operation', type=str, choices=['square', 'add', 'multiply', 'mean', 'variance'], required=True, help='FHE operation to perform')
        parser.add_argument('--value', type=float, nargs='+', help='Additional value(s) for operations')

    @classmethod
    def run(cls, cli):
        try:
            address = cli.config.address
            balance = cli.config.balance
            command = cli.config.command
            library = cli.config.library
            operation = cli.config.operation
            value = cli.config.value

            logger.info(f"Running FHE command for address: {address}")
            logger.info(f"Minimum balance: {balance}")
            logger.info(f"Command: {command}")
            logger.info(f"Using FHE library: {library}")
            logger.info(f"Operation: {operation}")

            if operation in ['add', 'multiply', 'mean', 'variance'] and value is None:
                raise ValueError("Value(s) must be provided for add, multiply, mean, or variance operations")

            if library == 'tenseal':
                cls.run_tenseal(command, operation, value)
            elif library == 'concrete':
                cls.run_concrete(command, operation, value)
            elif library == 'pyfhel':
                cls.run_pyfhel(command, operation, value)
            elif library == 'paillier':
                cls.run_paillier(command, operation, value)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise FHEError(f"FHE operation failed: {str(e)}")

    @staticmethod
    def run_tenseal(command: str, operation: str, value: list = None):
        try:
            context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
            context.global_scale = 2**40
            context.generate_galois_keys()

            x = ts.ckks_vector(context, [float(command)])
            if operation == 'square':
                result = x.square()
            elif operation == 'add':
                y = ts.ckks_vector(context, value)
                result = x + y
            elif operation == 'multiply':
                y = ts.ckks_vector(context, value)
                result = x * y
            elif operation == 'mean':
                y = ts.ckks_vector(context, value)
                result = (x + y.sum()) / (len(value) + 1)
            elif operation == 'variance':
                y = ts.ckks_vector(context, value)
                mean = (x + y.sum()) / (len(value) + 1)
                var = ((x - mean).square() + ((y - mean).square()).sum()) / (len(value) + 1)
                result = var
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            decrypted_result = result.decrypt()
            logger.info(f"TenSEAL result: {operation}({command}, {value}) = {decrypted_result}")
        except Exception as e:
            logger.error(f"TenSEAL operation failed: {str(e)}")
            raise FHEError(f"TenSEAL operation failed: {str(e)}")

    @staticmethod
    def run_concrete(command: str, operation: str, value: list = None):
        try:
            compiler = cnp.Compiler({"p_error": 1/2**10})
            
            if operation == 'square':
                @compiler.compile(inputset=[(3.14,)])
                def compute(x):
                    return x ** 2
            elif operation == 'add':
                @compiler.compile(inputset=[(3.14,)] * (len(value) + 1))
                def compute(*args):
                    return sum(args)
            elif operation == 'multiply':
                @compiler.compile(inputset=[(3.14,)] * (len(value) + 1))
                def compute(*args):
                    return np.prod(args)
            elif operation == 'mean':
                @compiler.compile(inputset=[(3.14,)] * (len(value) + 1))
                def compute(*args):
                    return sum(args) / len(args)
            elif operation == 'variance':
                @compiler.compile(inputset=[(3.14,)] * (len(value) + 1))
                def compute(*args):
                    mean = sum(args) / len(args)
                    return sum((x - mean) ** 2 for x in args) / len(args)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            inputs = [float(command)] + value
            circuit = compute.encrypt_run_decrypt(*inputs)
            logger.info(f"Concrete result: {operation}({command}, {value}) = {circuit}")
        except Exception as e:
            logger.error(f"Concrete operation failed: {str(e)}")
            raise FHEError(f"Concrete operation failed: {str(e)}")

    @staticmethod
    def run_pyfhel(command: str, operation: str, value: list = None):
        try:
            HE = pyfhel.Pyfhel()
            HE.contextGen(scheme='bfv', n=2**14, t_bits=20)
            HE.keyGen()
            HE.relinKeyGen()
            HE.rotateKeyGen()

            x = HE.encryptFrac(float(command))
            if operation == 'square':
                result = x * x
            elif operation == 'add':
                y = [HE.encryptFrac(v) for v in value]
                result = x + sum(y)
            elif operation == 'multiply':
                y = [HE.encryptFrac(v) for v in value]
                result = x * HE.cumProd(y)
            elif operation == 'mean':
                y = [HE.encryptFrac(v) for v in value]
                result = (x + sum(y)) / (len(value) + 1)
            elif operation == 'variance':
                y = [HE.encryptFrac(v) for v in value]
                mean = (x + sum(y)) / (len(value) + 1)
                var = ((x - mean)**2 + sum((yi - mean)**2 for yi in y)) / (len(value) + 1)
                result = var
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            decrypted_result = HE.decryptFrac(result)
            logger.info(f"Pyfhel result: {operation}({command}, {value}) = {decrypted_result}")
        except Exception as e:
            logger.error(f"Pyfhel operation failed: {str(e)}")
            raise FHEError(f"Pyfhel operation failed: {str(e)}")

    @staticmethod
    def run_paillier(command: str, operation: str, value: list = None):
        try:
            public_key, private_key = paillier.generate_paillier_keypair()
            x = public_key.encrypt(float(command))
            if operation == 'square':
                result = x * x
            elif operation == 'add':
                y = [public_key.encrypt(v) for v in value]
                result = x + sum(y)
            elif operation == 'multiply':
                result = x
                for v in value:
                    result *= v
            elif operation == 'mean':
                y = [public_key.encrypt(v) for v in value]
                result = (x + sum(y)) / (len(value) + 1)
            elif operation == 'variance':
                y = [public_key.encrypt(v) for v in value]
                mean = (x + sum(y)) / (len(value) + 1)
                var = ((x - mean)**2 + sum((yi - mean)**2 for yi in y)) / (len(value) + 1)
                result = var
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            decrypted_result = private_key.decrypt(result)
            logger.info(f"Paillier result: {operation}({command}, {value}) = {decrypted_result}")
        except Exception as e:
            logger.error(f"Paillier operation failed: {str(e)}")
            raise FHEError(f"Paillier operation failed: {str(e)}")

class FHEConfigCommand(BasedCommand):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--discovery_server', type=str, required=True, help='Discovery server address')
        parser.add_argument('--port', type=int, required=True, help='Port to listen on')
        parser.add_argument('--ollama_model', type=str, default='llama2', help='Ollama model to use')

    @classmethod
    def run(cls, cli):
        try:
            discovery_server = cli.config.discovery_server
            port = cli.config.port
            ollama_model = cli.config.ollama_model

            config = {
                "discovery_server": discovery_server,
                "port": port,
                "ollama_model": ollama_model
            }

            config_path = 'fhe_config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f)

            logger.info(f"FHE configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save FHE configuration: {str(e)}")
            raise FHEError(f"Failed to save FHE configuration: {str(e)}")

class FHEStartServerCommand(BasedCommand):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--config', type=str, default='fhe_config.json', help='Path to the configuration file')

    @classmethod
    def run(cls, cli):
        try:
            config_path = cli.config.config

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file {config_path} not found. Please run 'basedcli fhe config' first.")

            with open(config_path, 'r') as f:
                config = json.load(f)

            port = config['port']
            ollama_model = config.get('ollama_model', 'llama2')

            logger.info(f"Starting FHE server on port {port}")
            asyncio.get_event_loop().run_until_complete(cls.start_server(port, ollama_model))
        except Exception as e:
            logger.error(f"Failed to start FHE server: {str(e)}")
            raise FHEError(f"Failed to start FHE server: {str(e)}")

    @staticmethod
    async def start_server(port: int, ollama_model: str):
        async def handle_connection(websocket, path):
            try:
                async for message in websocket:
                    logger.info(f"Received message: {message}")
                    
                    # Process the message using Ollama
                    response = ollama.generate(model=ollama_model, prompt=message)
                    
                    # Encrypt the response before sending
                    encrypted_response = FHEStartServerCommand.encrypt_response(response['response'])
                    
                    # Send the encrypted Ollama response back
                    await websocket.send(json.dumps(encrypted_response))
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {str(e)}")

        try:
            server = await websockets.serve(handle_connection, "localhost", port)
            logger.info(f"Server started on ws://localhost:{port}")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}")
            raise FHEError(f"Failed to start WebSocket server: {str(e)}")

    @staticmethod
    def encrypt_response(response: str) -> Dict[str, Any]:
        # This is a placeholder for actual FHE encryption
        # In a real implementation, you would use one of the FHE libraries to encrypt the response
        # For demonstration, we'll use TenSEAL for actual FHE encryption
        try:
            context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
            context.global_scale = 2**40
            context.generate_galois_keys()

            # Convert the response string to a list of ASCII values
            ascii_values = [ord(char) for char in response]
            
            # Encrypt the ASCII values
            encrypted_vector = ts.ckks_vector(context, ascii_values)
            
            # Serialize the encrypted vector
            serialized_vector = encrypted_vector.serialize()

            return {
                "encrypted_data": serialized_vector.hex(),
                "context": context.serialize().hex()
            }
        except Exception as e:
            logger.error(f"FHE encryption failed: {str(e)}")
            raise FHEError(f"FHE encryption failed: {str(e)}")
