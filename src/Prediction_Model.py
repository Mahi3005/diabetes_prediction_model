import nada_numpy as na
from nada_ai.linear_model import LinearRegression

def nada_main():
    # Set precision
    na.set_log_scale(32)

    # Step 1: Use Nada NumPy wrapper to create "Party0" and "Party1"
    parties = na.parties(2)

    # Step 2: Instantiate linear regression object
    feature_count = 8
    my_model = LinearRegression(in_features=feature_count)

    # Step 3: Load model weights from Nillion network
    my_model.load_state_from_network("my_model", parties[0], na.SecretRational)

    # Step 4: Load input data for inference (provided by Party1)
    my_input = na.array((feature_count,), parties[1], "my_input", na.SecretRational)

    # Step 5: Compute inference
    result = my_model.forward(my_input)

    # Step 6: Produce output for Party1
    return na.output(result, parties[1], "my_output")

