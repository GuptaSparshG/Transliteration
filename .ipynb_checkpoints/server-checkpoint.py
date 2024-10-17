from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import boto3
from io import BytesIO
from ai4bharat.transliteration import XlitEngine
import uvicorn

# Initialize the S3 client
s3 = boto3.client('s3')

app = FastAPI()

class TransliterationRequest(BaseModel):
    csv: str
    attributes: dict  # Dictionary of column name and target language pairs

def read_s3_file(s3_uri):
    """Download the file from S3 and load it into a pandas DataFrame."""
    bucket_name, key = parse_s3_uri(s3_uri)

    # Download the file from S3
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    file_content = obj['Body'].read()

    # Determine if it's a CSV or Excel file
    if key.endswith('.csv'):
        df = pd.read_csv(BytesIO(file_content))
    elif key.endswith('.xlsx'):
        df = pd.read_excel(BytesIO(file_content))
    else:
        raise ValueError("The file format is not supported. Please provide a CSV or Excel file.")
    
    return df

def write_s3_file(df, s3_uri):
    """Write the DataFrame back to S3 after dropping specific columns."""
    # Drop columns that end with 'tts', 'vc', or 'ls'
    df = df.loc[:, ~df.columns.str.endswith(('tts', 'vc', 'ls'))]

    bucket_name, key = parse_s3_uri(s3_uri)

    # Save the DataFrame to a BytesIO object
    output = BytesIO()
    if key.endswith('.csv'):
        df.to_csv(output, index=False)
    elif key.endswith('.xlsx'):
        df.to_excel(output, index=False, engine='xlsxwriter')
    
    output.seek(0)  # Reset the buffer to the beginning

    # Upload the file back to S3
    s3.put_object(Bucket=bucket_name, Key=key, Body=output.getvalue())

    return f"s3://{bucket_name}/{key}"

def parse_s3_uri(s3_uri):
    """Parse an S3 URI into bucket and key."""
    if s3_uri.startswith("s3://"):
        path_parts = s3_uri.replace("s3://", "").split("/", 1)
        return path_parts[0], path_parts[1]
    else:
        raise ValueError("Invalid S3 URI")

@app.post("/transliterate/")
def transliterate_names(request: TransliterationRequest):
    """Perform transliteration on names from the input file and upload the result to S3."""
    input_s3_uri = request.csv
    attributes = request.attributes

    # Load the input file from S3
    df = read_s3_file(input_s3_uri)

    # Print the initial DataFrame to verify input
    print("Initial DataFrame:")
    print(df)

    # Mapping of user-friendly language names to language codes
    language_dict = {
        'Bengali': 'bn',
        'Urdu': 'ur',
        'Nepali': 'ne',
        'Goan': 'gom',
        'Telugu': 'te',
        'Sinhalese': 'si',
        'Maithili': 'mai',
        'Gujarati': 'gu',
        'Manipuri': 'mni',
        'Marathi': 'mr',
        'Tamil': 'ta',
        'Sindhi': 'sd',
        'Assamese': 'as',
        'Sanskrit': 'sa',
        'Hindi': 'hi',
        'Malayalam': 'ml',
        'Kashmiri': 'ks',
        'Bodo': 'brx',
        'Odia': 'or',
        'Kannada': 'kn',
        'Punjabi': 'pa',
        'English': None  # English does not require transliteration
    }

    # Dictionary to store previously transliterated names to avoid duplicating work
    transliterated_cache = {}

    # Create a dictionary to hold names for each language
    transliteration_dict = {}

    # Group names by language using language codes
    for column, lang_name in attributes.items():
        lang_code = language_dict.get(lang_name)

        for index, row in df.iterrows():
            name = row[column]

            # Skip rows with missing values
            if pd.isna(name):
                continue

            # If English, just copy the original value into the transliteration column
            if lang_name == 'English':
                translit_column_name = f'Transliteration_{column}_en'
                status_column_name = f'Status_{column}_en'  # Define the corresponding status column name
                if translit_column_name not in df.columns:
                    col_idx = df.columns.get_loc(column) + 1
                    df.insert(col_idx, translit_column_name, name)
                    df.insert(col_idx + 1, status_column_name, '')  # Insert the Status column
                else:
                    df[translit_column_name] = name  # Fill the existing column with the original name
                df[status_column_name] = "Transliterated"  # Set status for English
                continue

            # If name has already been transliterated, reuse it
            if (name, lang_code) in transliterated_cache:
                translit_text = transliterated_cache[(name, lang_code)]
                translit_column_name = f'Transliteration_{column}_{lang_name}'  # Update to use full language name
                status_column_name = f'Status_{column}_{lang_name}'  # Define the corresponding status column name
                if translit_column_name not in df.columns:
                    col_idx = df.columns.get_loc(column) + 1
                    df.insert(col_idx, translit_column_name, '')
                    df.insert(col_idx + 1, status_column_name, '')  # Insert the Status column
                df.loc[df[column] == name, translit_column_name] = translit_text
                df.loc[df[column] == name, status_column_name] = "Transliterated"  # Set status for cached names
                continue

            # Add names to be transliterated if not in cache
            if lang_code not in transliteration_dict:
                transliteration_dict[lang_code] = []  # Initialize the list if the language is not yet in the dict
            transliteration_dict[lang_code].append((index, name, column))  # Append tuple (index, name, column name)

    # Initialize a transliteration engine for each unique language that is supported
    engines = {lang_code: XlitEngine(lang_code, beam_width=10) for lang_code in transliteration_dict.keys()}

    # Transliterate names in batch for languages other than English
    for lang_code, name_list in transliteration_dict.items():
        names_to_transliterate = [name for _, name, _ in name_list]

        print(f"\nTransliterating {len(names_to_transliterate)} names for language '{lang_code}': {names_to_transliterate}")

        try:
            # Batch transliteration
            results = [engines[lang_code].translit_sentence(name) for name in names_to_transliterate]

            # Check results validity
            print(f"Results for {lang_code}: {results}")

            for (index, original_name, name_column), translit_result in zip(name_list, results):
                try:
                    # Extract the transliterated text for the corresponding language code
                    translit_text = translit_result.get(lang_code, original_name)

                    # Define the new column name for the transliteration using full language name
                    translit_column_name = f'Transliteration_{name_column}_{lang_code}'  # Use full language name
                    status_column_name = f'Status_{name_column}_{lang_code}'  # Define the corresponding status column name

                    # Insert the new column after the original column if it doesn't exist
                    if translit_column_name not in df.columns:
                        col_idx = df.columns.get_loc(name_column) + 1
                        df.insert(col_idx, translit_column_name, '')
                        df.insert(col_idx + 1, status_column_name, '')  # Insert the Status column

                    # Update all instances of the name with the transliterated text
                    df.loc[df[name_column] == original_name, translit_column_name] = translit_text
                    df.loc[df[name_column] == original_name, status_column_name] = "Pending"  # Set status

                    # Cache the transliterated text to avoid duplicating work
                    transliterated_cache[(original_name, lang_code)] = translit_text

                    print(f"Transliterated '{original_name}' to '{translit_text}' in index {index}.")
                except Exception as e:
                    # Catch any errors for individual names to ensure the process continues
                    print(f"Error while processing '{original_name}' in index {index}: {e}")

        except Exception as ex:
            print(f"Error in transliterating names for language '{lang_code}': {ex}")

    # Save the updated DataFrame back to S3 after dropping unnecessary columns
    output_s3_uri = write_s3_file(df, input_s3_uri)  # You can use a different output path if required
    print(f"Transliteration completed and saved to {output_s3_uri}.")

    return {"output_s3_uri": output_s3_uri}

# Entry point to run the server using the python keyword
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 
