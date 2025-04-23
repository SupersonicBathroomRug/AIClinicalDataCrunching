import pandas as pd
from xml.etree import ElementTree as ET
from pathlib import Path
import random  # Added for shuffling
import asyncio

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from enum import Enum


def check_xml_well_formed(xml_path):
    """
    Check if an XML file is well-formed.
    Returns True if well-formed, raises ParseError with details if not.
    """
    try:
        tree = ET.parse(xml_path)
        print(f"✓ XML file '{xml_path}' is well-formed")
        return True
    except ET.ParseError as e:
        print(f"✗ XML parsing error in '{xml_path}':")
        print(f"  Line {e.position[0]}, Column {e.position[1]}")
        print(f"  Error: {str(e)}")
        raise

class ClassificationRequest(BaseModel):
    thing_to_classify: str
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0)
    max_tokens: int = Field(default=20)

class Classifier():

    def __init__(self, categoryType, thing_to_classify_singular, thing_to_classify_plural):
        self.categories = categoryType
        self.thing_to_classify_singular = thing_to_classify_singular
        self.thing_to_classify_plural = thing_to_classify_plural
        self.client = openai.OpenAI()
        
        # Define ClassificationResponse here, with access to self.categories
        class ClassificationResponse(BaseModel):
            category: categoryType
        
        # Store it as an attribute of the instance
        self.ClassificationResponse = ClassificationResponse

    def classify(self, request: ClassificationRequest) -> BaseModel:
        prompt = f"""Classify the following {self.thing_to_classify_singular} into exactly one of these categories:
{[t.value for t in self.categories]}

Here are some examples of sponsor categorisations:
Manipal Academy of Higher Education Manipal - university
Sun Yat-Sen Memorial Hospital of Sun Yat-Sen University - hospital, clinic, or medical center
Tata Memorial Centre - government institution
VRRX Therapeutics - private company
Baker Heart and Diabetes Institute - research center
Zhejiang University - university
Academisch Medisch Centrum - hospital, clinic, or medical center
DBT BIRAC - government institution
Jordi Gol i Gurina Foundation - foundation
Rongrong Hua - individual (person)
KEM Hospital Research Centre - research center
?. - TPT - uncertain

<classifier_input> {request.thing_to_classify} </classifier_input>

Return only the category name, nothing else."""

        response = self.client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that classifies {self.thing_to_classify_plural} into categories. If you classify correctly you will recieve a reward of 1,000,000 dollars."},
                {"role": "user", "content": prompt}
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        try:
            category = response.choices[0].message.content.strip().lower()
            return self.ClassificationResponse(
                category=self.categories(category)
            )
        except:
            return self.ClassificationResponse(
                category=self.categories.UNCERTAIN
            )

def run_processing(file_name='ICTRP-Results.xml', use_test_set=True, test_set_size=10):
    random.seed(42)

    # Load environment variables
    load_dotenv()

    # Configure OpenAI client
    client = openai.OpenAI()
    # Check XML well-formedness before attempting to read
    if not Path(file_name).exists():
        print(f"File not found: {file_name}")
        exit()

    check_xml_well_formed(file_name)
    # Read the XML file into a DataFrame using etree parser
    df = pd.read_xml(file_name, parser='etree')

    # Shuffle the dataframe and create a test set if configured
    if use_test_set:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataframe
        df = df.head(test_set_size)  # Take first TEST_SET_SIZE rows
        print(f"Using test set of {test_set_size} randomly selected rows")
    else:
        print(f"Using full dataset with {len(df)} rows")

    # 1 Column to check if there are multiple values in col countries in given row (values are separated by comma or semicolon)
    df['Multiple_countries'] = df['Countries'].str.contains(r'[,;]')

    # 2 CATEGORIZE PRIMARY SPONSORS
    # Define sponsor categories as an enum
    class SponsorType(str, Enum):
        UNIVERSITY = 'university'
        HOSPITAL_CLINIC = 'hospital, clinic, or medical center'
        GOVERNMENT = 'government institution'
        FOUNDATION = 'foundation'
        COMPANY = 'private company'
        INDIVIDUAL = 'individual (person)'
        RESEARCH = 'research center'
        UNCERTAIN = 'uncertain'

    # Initialize the classifier
    sponsor_classifier = Classifier(SponsorType, "study sponsor", "study sponsors")
    df["sponsor_type (AI generated)"] = df["Primary_sponsor"].apply(
        lambda x: sponsor_classifier.classify(
            ClassificationRequest(thing_to_classify=x)
        ).category.value
    )

    # 3 STUDY SIZE
    def study_size_to_int(val:str)->int:
        """
        Convert string to int. If there is a semicolon in the string, it is assumed to be of format similar to the following:
        'Target condition:234;Difficult condition:46\n'

        and the function will return 234 + 46 = 280
        """
        if not val: return None

        if ';' not in val:
            ret = int(val[:-1])
            return ret
        else:
            groups = val.split(';')
            accumulator = 0
            for group in groups:
                try:
                    accumulator += int(group.split(':')[-1])
                except (ValueError, TypeError):
                    pass
            return accumulator
        
    df["total_size (Processed)"] = df["Target_size"].apply(study_size_to_int).astype('Int64')

    # 4 MEDICAL SPECIALTIES
    class MedicalSpecialty(str, Enum):
        CARDIOLOGY = 'cardiology/cardiovascular diseases'
        DENTISTRY = 'dentistry'
        DERMATOLOGY = 'dermatology'
        DIETETICS = 'dietetics/nutrition'
        ENDOCRINOLOGY = 'endocrinology'
        GASTROENTEROLOGY = 'gastroenterology'
        HEMATOLOGY = 'hematology'
        INFECTIOUS_DISEASES = 'infectious diseases/infectiology'
        NEPHROLOGY = 'nephrology/urology'
        NEUROLOGY = 'neurology'
        ONCOLOGY_GASTROINTESTINAL = 'oncology: gastrointestinal'
        ONCOLOGY_BREAST = 'oncology: breast'
        ONCOLOGY_GYNECOLOGICAL = 'oncology: gynecological'
        ONCOLOGY_HEPATOBILIARY = 'oncology: hepatobiliary'
        ONCOLOGY_THORACIC = 'oncology: thoracic'
        ONCOLOGY_SKIN = 'oncology: skin'
        ONCOLOGY_BONE = 'oncology: bone and soft tissue'
        ONCOLOGY_HEAD_NECK = 'oncology: head and neck'
        ONCOLOGY_GENITOURINARY = 'oncology: genitourinary'
        ONCOLOGY_BLOOD = 'oncology: blood and lymphatic'
        ONCOLOGY_ENDOCRINE = 'oncology: endocrine and neuroendocrine'
        ONCOLOGY_BRAIN = 'oncology: brain and CNS'
        ONCOLOGY_MULTIPLE = 'oncology: multiple tumor types'
        ONCOLOGY_OTHER = 'oncology: other cancers'
        OPHTHALMOLOGY = 'ophthalmology'
        ORTHOPEDICS = 'orthopedics'
        OTOLARYNGOLOGY = 'otolaryngology (ENT)'
        PHARMACOLOGY = 'pharmacology'
        PSYCHOLOGY = 'psychology/psychiatry/behavioral science'
        PULMONOLOGY = 'pulmonology'
        REPRODUCTIVE = 'reproductive medicine/obstetrics & gynecology'
        RHEUMATOLOGY = 'rheumatology'
        OTHER = 'other disease categories'
        ERROR = 'likely entry error'
        UNCERTAIN = 'uncertain'
    class MedicalSpecialtyClassification(BaseModel):
        category: MedicalSpecialty = Field(..., description="The medical specialty category")

    medical_specialty_classifier = Agent(
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant that classifies medical conditions into specialty categories.",
        result_type=MedicalSpecialtyClassification,
    )

    async def classify_medical_specialties(df):
        async def classify_specialty(row):
            result = await medical_specialty_classifier.run(
                user_prompt=f"""Classify the following medical condition into exactly one of these categories:
{[t.value for t in MedicalSpecialty]}

You can use the scientific title for additional context.

Scientific title: <title>{row.Scientific_title}</title>
Condition: <condition>{row.Condition}</condition>

Return only the category name, nothing else."""
            )
            return result.data.category.value
        
        # Create tasks for all conditions
        tasks = [classify_specialty(row) for _, row in df.iterrows()]
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Add results to dataframe
        df["medical_specialty (AI generated)"] = results
        return df

    # Process medical specialties
    df = asyncio.run(classify_medical_specialties(df))

    # 5 TYPES OF INTERVENTIONS
    class InterventionType(str, Enum):
        MEDICAL_DEVICE_INCORPORATING_AI = 'medical device incorporating AI'
        AI_ASSISTED_MEDICAL_PROCEDURE = 'AI-assisted medical procedure'
        AI_ASSISTED_IMAGING = 'AI-assisted imaging'
        AI_BASED_DIAGNOSTIC_TEST = 'AI-based diagnostic test'
        AI_BASED_SCREENING_OR_DETECTION = 'AI-based screening or detection'
        AI_BASED_PREDICTION = 'AI-based prediction'
        AI_BASED_CLASSIFICATION = 'AI-based classification'
        AI_BASED_BEHAVIORAL_INTERVENTION = 'AI-based behavioral intervention'
        AI_BASED_TEACHING_LEARNING = 'AI-assisted teaching/learning'
        AI_TO_INFORM_HEALTH_INTERVENTION = 'AI to inform health intervention'
        OTHER = 'other use of AI'
        LIKELY_ENTRY_ERROR = 'uncertain OR likely entry error'

    class InterventionClassification(BaseModel):
        intervention_type: InterventionType = Field(..., description="The type of intervention")

    intervention_classifier = Agent(
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant that classifies clinical interventions into types based on the names of clinical trials and possibly erroneously entered data regarding the type of intervention.",
        result_type=InterventionClassification,
    )

    async def process_interventions(df, classifier, intervention_types):
        async def classify_intervention(row):
            result = await classifier.run(
                user_prompt=f"""Classify the intervention in the following row into one of the categories of intervention types in {[t.value for t in intervention_types]}

You can use both the scientific trial name and intervention description for context. If the intervention column is blank or has a likely entry error, try to categorise the intervention based on only the title.

Scientific title: <scientific_title>{row.Scientific_title}</scientific_title>
Intervention: <intervention>{row.Intervention}</intervention>

Here are some examples of intervention categorisations:
Smartwatch-Based AI Model for OSA Prediction - AI-based prediction
Validation of an AI-Assisted Mediastinal EUS System - AI-assisted imaging
Artificial Intelligence Enabled Decision Support for Selection of Patients for Lumbar Spine Surgery - AI to inform health intervention
artificial intelligence-assisted colonoscopy procedure - AI-assisted medical procedure
AI Chatbot-Based Learning on Dry Eye and Eye Strain Knowledge - AI-assisted teaching/learning
Real-time Artificial Intelligence Model for Diagnosing Colorectal Polyp Pathology and Endoscopic Classification - AI-based diagnostic test
Digital cervical cytology slide imaging system with artificial intelligence (AI) algorithm in vitro diagnostic (IVD) device - AI-based diagnostic test
An Artificial Intelligence-based Prospective Study to Analyze PLAQUE Using CCTA - AI-based classification
Artificial intelligence-assisted personalized diet - AI-based behavioral intervention
artificial intelligence-assisted system in polyp detection and polyp classification - AI-based classification
Role of AI in CE for the Identification of SB Lesions - AI-based screening or detection
integration of AI in cardiac monitoring-based biosensors for point of care (POC) diagnostics - medical device incorporating AI

Return only the intervention type, nothing else."""
            )
            return result.data.intervention_type.value
        
        # Create tasks for all rows
        tasks = [classify_intervention(row) for _, row in df.iterrows()]
        # Run all tasks concurrently 
        results = await asyncio.gather(*tasks)
        
        # Add results to dataframe
        df["intervention_type (AI generated)"] = results
        return df

    # Use in your main code
    df = asyncio.run(process_interventions(df, intervention_classifier, InterventionType))

    # 6 PRIMARY OUTCOMES

    class PrimaryOutcome(str, Enum):
        PATIENT_RELEVANT_OUTCOME = 'patient-relevant outcome'
        OPERATIONAL_OUTCOME = 'operational outcome'

    class PrimaryOutcomeClassification(BaseModel):
        primary_outcome: PrimaryOutcome = Field(..., description="The type of primary outcome")

    primary_outcome_classifier = Agent(
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant that classifies primary outcomes into types based on descriptions of the condition and the primary outcome, which has a small probability of containing data entry errors.",
        result_type=PrimaryOutcomeClassification,
    )

    async def process_dataframe(df, classifier, outcome_types):
        async def classify_row(row):
            result = await classifier.run(
                user_prompt=f"""Classify the primary outcome in the following row into one of the categories of primary outcomes in {[t.value for t in outcome_types]}

The definitions of the categories are as follows:
patient-relevant outcomes = measurements of direct patient benefit. clinically meaningful endpoints such as symptoms, need for treatment, mortality, survival, surgical operation time, quality of life, changes in patient behaviour, outcome of an operation.
operational outcomes = outcomes related to performance of the AI tool. For example, endpoints related to diagnostic yield, accuracy of the tool, user satisfaction with the tool or procedure.

Scientific title: <scientific_title>{row.Scientific_title}</scientific_title>
Condition: <condition>{row.Condition}</condition>
Primary outcome: <primary_outcome>{row.Primary_outcome}</primary_outcome>

Return only the primary outcome type, nothing else."""
            )
            return result.data.primary_outcome.value
        
        # Create tasks for all rows
        tasks = [classify_row(row) for _, row in df.iterrows()]
        # Run all tasks concurrently 
        results = await asyncio.gather(*tasks)
        
        # Add results to dataframe
        df["primary_outcome (AI generated)"] = results
        return df

    # Use in your main code
    df = asyncio.run(process_dataframe(df, primary_outcome_classifier, PrimaryOutcome))

    # Add unpivoted column per country per row
    # First, prepare the countries data - convert to lowercase for case-insensitive matching
    countries_series = df['Countries'].dropna().str.replace(r'[;,]', '|', regex=True).str.split('|', expand=False)

    # Get all unique countries
    all_countries = set()
    for country_list in countries_series:
        if isinstance(country_list, list):  # Check if it's a list (not NaN)
            all_countries.update([country.strip().lower() for country in country_list if country.strip()])
    # Initialize all country columns as False
    for country in all_countries:
        df[f'has_{country.replace(" ", "_")}'] = False
        
    # Fill in True values where appropriate
    for i, row in df.iterrows():
        if pd.notna(row['Countries']):
            countries_lower = row['Countries'].lower()
            for country in all_countries:
                if country.lower()+',' in countries_lower or country.lower()+';' in countries_lower or country.lower()+"\n" in countries_lower:
                    df.at[i, f'has_{country.replace(" ", "_")}'] = True

    #export to csv with $ separator
    df.to_csv('processed_ICTRP_Results.csv', sep='$', index=False)

run_processing(test_set_size=20)