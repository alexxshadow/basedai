import json
import numpy as np  # If you don't need NumPy, you can remove this import

def run_paillier_operation(operation, data):
    from phe import paillier
    
    # Deserialize the public key and encrypted vector
    public_key = paillier.PaillierPublicKey(n=int(data['public_key']))
    vector = [paillier.EncryptedNumber(public_key, int(x)) for x in data['vector']]
    
    if operation == 'square':
        result = [x * x for x in vector]
    elif operation == 'add':
        result = [x + x for x in vector]
    elif operation == 'multiply':
        result = [x * x for x in vector]
    elif operation == 'mean':
        result = sum(vector) / len(vector)
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return [str(x.ciphertext()) for x in result]

if __name__ == "__main__":
    with open('/app/input_data.json', 'r') as f:
        input_data = json.load(f)
    
    library = input_data['library']
    operation = input_data['operation']
    data = input_data['data']
    
    if library == 'paillier':
        result = run_paillier_operation(operation, data)
    else:
        raise ValueError(f"Unsupported library: {library}")
    
    with open('/app/output_data.json', 'w') as f:
        json.dump({'result': result}, f)
