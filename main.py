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
from nillion_utils import compute, store_program, store_secrets, get_user_id_by_seed
from nada_ai.client import SklearnClient
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load environment variables from a .env file
load_dotenv()

# Constants
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 30  # seconds

async def setup_nillion():
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
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age']
        X = data[features].values
        y = data['Outcome'].values

        # Train the Model
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

        return client_1, payments_wallet, payments_client, program_id, cluster_id, party_names, party_id_0, party_id_1, model_store_id
    except Exception as e:
        st.error(f"Failed to set up the secure computation environment. Error: {str(e)}")
        raise

async def predict_diabetes(client_1, payments_wallet, payments_client, program_id, cluster_id, party_names, party_id_0,
                           party_id_1, model_store_id, new_patient_data):
    try:
        new_patient_array = np.array(new_patient_data)
        my_input = na_client.array(new_patient_array, "my_input", na.SecretRational)
        input_secrets = nillion.NadaValues(my_input)

        # Set up the compute bindings for the parties
        compute_bindings = nillion.ProgramBindings(program_id)
        compute_bindings.add_input_party(party_names[0], party_id_0)
        compute_bindings.add_input_party(party_names[1], party_id_1)
        compute_bindings.add_output_party(party_names[1], party_id_1)

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
        output = na_client.float_from_rational(inference_result["my_output"])
        return output
    except asyncio.TimeoutError:
        st.error("The prediction operation timed out. Please try again.")
        raise
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        raise

async def retry_operation(operation, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(operation(), timeout=30)  # 30-second timeout
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise
            delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
            st.warning(f"Operation timed out. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
            st.warning(f"Operation failed. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)

def clear_results():
    if 'prediction' in st.session_state:
        del st.session_state['prediction']
    if 'risk_level' in st.session_state:
        del st.session_state['risk_level']

def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

    st.title("🩺 Diabetes Risk Prediction Model")
    st.write("Enter patient data to predict diabetes risk. This model uses secure computation to protect your data.")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.3,
                                            format="%.3f")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)

    if st.button("Predict Risk", key="predict_button"):
        clear_results()  # Clear previous predictions
        new_patient_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

        if 'nillion_setup' not in st.session_state:
            with st.spinner("Setting up secure computation environment..."):
                try:
                    st.session_state.nillion_setup = asyncio.run(retry_operation(setup_nillion))
                except Exception as e:
                    st.error(f"Failed to set up the secure computation environment. Please try again later. Error: {str(e)}")
                    return

        with st.spinner("Performing secure prediction..."):
            try:
                client_1, payments_wallet, payments_client, program_id, cluster_id, party_names, party_id_0, party_id_1, model_store_id = st.session_state.nillion_setup
                prediction = asyncio.run(retry_operation(lambda: predict_diabetes(
                    client_1, payments_wallet, payments_client, program_id, cluster_id, party_names,
                    party_id_0, party_id_1, model_store_id, new_patient_data)))
                st.session_state['prediction'] = prediction
                st.session_state['risk_level'] = "High" if prediction > 0.6 else "Moderate" if prediction > 0.4 else "Low"
            except Exception as e:
                st.error(f"Failed to perform the prediction. Please try again later. Error: {str(e)}")
                return

    if 'prediction' in st.session_state and 'risk_level' in st.session_state:
        st.subheader("Prediction Result")
        color = "red" if st.session_state['risk_level'] == "High" else "orange" if st.session_state['risk_level'] == "Moderate" else "green"

        st.markdown(f"<h3 style='color: {color};'>Risk Level: {st.session_state['risk_level']}</h3>",
                    unsafe_allow_html=True)
        st.write(f"Prediction value: {st.session_state['prediction']:.4f}")

        st.info(
            "Please note that this prediction is based on a machine learning model and should not be considered as a medical diagnosis. Always consult with a healthcare professional for proper medical advice.")

    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a secure computation framework to predict diabetes risk "
        "while keeping your data private. The prediction is made using a linear "
        "regression model trained on the Pima Indians Diabetes Database."
    )
    st.sidebar.warning(
        "Disclaimer: This tool is for educational purposes only and should not "
        "be used as a substitute for professional medical advice, diagnosis, or treatment."
    )

    if st.sidebar.button("Clear Results and Reset"):
        clear_results()
        if 'nillion_setup' in st.session_state:
            del st.session_state['nillion_setup']
        st.rerun()

if __name__ == "__main__":
    main()
