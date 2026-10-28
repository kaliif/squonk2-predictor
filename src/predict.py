import argparse
import pickle
import pandas as pd

import descriptor_generator
from descriptor_generator import run as calculate_mordred

temp_desc_file = 'molecules_calcd.csv'

def run(
        sk_model,
        molecules,
):

    with open(sk_model, 'rb') as f:
        model = pickle.load(f)

        
    print(model)
    print(type(model))
    print(dir(model))    

    calculate_mordred(
        molecules,
        temp_desc_file,
        delimiter=',',
        write_header=True,


        
    )

    descriptors = pd.read_csv(temp_desc_file)
    print(descriptors)

    predictions = model.predict(descriptors)
    print(predictions)
    print(predictions.shape)

    print('asdopughas')
    
    
    
def main():
    # Example:
    # python src/predict.py
    #     --read-header\
    #     --outfile=outfile.json\
    #     --infile=data/caco2_mordred_filtered_scaled.smi\
    #     --bootstrap=True,False\
    #     --ccp_alpha=0.0,0.1,0.5

    parser = argparse.ArgumentParser(description="Prepare data")

    parser.add_argument("--model", required=True, help="SciKit .jmodel file")
    parser.add_argument("--molecules", required=True, help="Molecules to predict")

    args = parser.parse_args()

    run(
        sk_model=args.model,
        molecules=args.molecules,
    )


if __name__ == "__main__":
    main()    
