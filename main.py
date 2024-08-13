import os
import asyncio
import numpy as np
import pandas as pd
import nada_numpy as na
import nada_numpy.client as na_client
import py_nillion_client as nillion
from py_nillion_client import NodeKey, UserKey
from nillion_python_helpers import create_nillion_client, create_payments_config
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_utils import compute, store_program, store_secrets, get_user_id_by_seed  # local helper file
from nada_ai.client import SklearnClient

# Your code where SklearnClient is used


# Load environment variables from a .env file
load_dotenv()

async def main():
    try:
        # Set log scale to match the precision set in the nada program
        na.set_log_scale(32)
        program_name = "Prediction_Model"
        program_mir_path = f"./target/{program_name}.nada.bin"

        cluster_id = os.getenv("NILLION_CLUSTER_ID")
        grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
        chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")

        # Create 2 parties - Party0 and Party1
        party_names = na_client.parties(2)

        # Create NillionClient for Party0, storer of the model
        seed_0 = 'seed-party-model'
        userkey_party_0 = nillion.UserKey.from_seed(seed_0)
        nodekey_party_0 = nillion.NodeKey.from_seed(seed_0)
        client_0 = create_nillion_client(userkey_party_0, nodekey_party_0)
        party_id_0 = client_0.party_id
        user_id_0 = client_0.user_id

        # Create NillionClient for Party1
        seed_1 = 'seed-party-input'
        userkey_party_1 = nillion.UserKey.from_seed(seed_1)
        nodekey_party_1 = nillion.NodeKey.from_seed(seed_1)
        client_1 = create_nillion_client(userkey_party_1, nodekey_party_1)
        party_id_1 = client_1.party_id
        user_id_1 = client_1.user_id

        # Configure payments
        payments_config = create_payments_config(chain_id, grpc_endpoint)
        payments_client = LedgerClient(payments_config)
        payments_wallet = LocalWallet(
            PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))),
            prefix="nillion",
        )

        # Party0 stores the linear regression Nada program
        program_id = await store_program(
            client_0,
            payments_wallet,
            payments_client,
            user_id_0,
            cluster_id,
            program_name,
            program_mir_path,
        )

        # Load the transformed dataset
        data = pd.read_csv('./diabetes-transformed.csv')
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        X = data[features].values
        y = data['Outcome'].values

        # Train the Model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        # Create SklearnClient with nada-ai
        model_client = SklearnClient(model)

        # Party0 creates a secret
        model_secrets = nillion.NadaValues(model_client.export_state_as_secrets("my_model", na.SecretRational))

        # Create permissions for model_secrets
        permissions_for_model_secrets = nillion.Permissions.default_for_user(user_id_0)
        allowed_user_ids = [user_id_1] + [get_user_id_by_seed(f"inference_{i}") for i in range(1, 4)]
        permissions_dict = {user: {program_id} for user in allowed_user_ids}
        permissions_for_model_secrets.add_compute_permissions(permissions_dict)

        # Party0 stores the model as a Nillion Secret
        model_store_id = await store_secrets(
            client_0,
            payments_wallet,
            payments_client,
            cluster_id,
            model_secrets,
            1,
            permissions_for_model_secrets,
        )

        # Party1 creates the new input secret
        new_patient_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]  # Example input
        new_patient_array = np.array(new_patient_data)
        my_input = na_client.array(new_patient_array, "my_input", na.SecretRational)
        input_secrets = nillion.NadaValues(my_input)

        # Set up the compute bindings for the parties
        compute_bindings = nillion.ProgramBindings(program_id)
        compute_bindings.add_input_party(party_names[0], party_id_0)
        compute_bindings.add_input_party(party_names[1], party_id_1)
        compute_bindings.add_output_party(party_names[1], party_id_1)

        print(f"Computing using program {program_id}")
        print(f"Use secret store_id: {model_store_id}")

        # Party1 performs blind computation that runs inference and returns the result
        inference_result = await compute(
            client_1,
            payments_wallet,
            payments_client,
            program_id,
            cluster_id,
            compute_bindings,
            [model_store_id],
            input_secrets,
            verbose=True,
        )

        # Rescale the obtained result
        outputs = [na_client.float_from_rational(inference_result["my_output"])]
        print(f"🙈 The rescaled result computed by the {program_name} Nada program is {outputs[0]}")
        expected = model.predict(new_patient_array.reshape(1, -1))
        print(f"👀 The expected result computed by sklearn is {expected[0]}")

        print(f"""
        Features of new input data:
            Pregnancies: {new_patient_data[0]}
            Glucose level: {new_patient_data[1]}
            Blood Pressure: {new_patient_data[2]}
            Skin Thickness: {new_patient_data[3]}
            Insulin level: {new_patient_data[4]}
            BMI: {new_patient_data[5]}
            Diabetes Pedigree Function: {new_patient_data[6]}
            Age: {new_patient_data[7]}
        """)
        print(f"The predicted outcome for this patient is: {'Positive' if outputs[0] > 0.5 else 'Negative'}")

        return inference_result

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
