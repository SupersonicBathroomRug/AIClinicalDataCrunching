# prompts.py

class ClassifierPrompts:
    GENERIC_CLASSIFIER = """Classify the following {thing_to_classify_singular} into exactly one of these categories:
{categories}

<classifier_input> {input_text} </classifier_input>

Return only the category name, nothing else."""

    GENERIC_SYSTEM_MESSAGE = """You are a helpful assistant that classifies {thing_to_classify_plural} into categories. If you classify correctly you will recieve a reward of 1,000,000 dollars."""

class MedicalSpecialtyPrompts:
    CLASSIFY_SPECIALTY = """Classify the following medical condition into exactly one of these categories:
{categories}

<condition>{condition}</condition>

Return only the category name, nothing else."""

    SYSTEM_MESSAGE = "You are a helpful assistant that classifies medical conditions into specialty categories."

class InterventionPrompts:
    CLASSIFY_INTERVENTION = """Classify the intervention in the following row into one of the categories of intervention types in {categories}

The following is the name of the trial:
<trial_name> {trial_name} </trial_name>

The following is how the intervention is described in the trial, note that this may contain data entry errors:
<intervention> {intervention} </intervention>

Return only the intervention type, nothing else."""

    SYSTEM_MESSAGE = "You are a helpful assistant that classifies clinical interventions into types based on the names of clinical trials and possibly erroneously entered data regarding the type of intervention."

class PrimaryOutcomePrompts:
    CLASSIFY_OUTCOME = """Classify the primary outcome in the following row into one of the categories of primary outcomes in {categories}

The definitions of the categories are as follows:
patient-relevant outcomes = measurements of direct patient benefit. clinically meaningful endpoints such as symptoms, need for treatment, mortality, survival, surgical operation time, quality of life
operational outcomes = endpoints related to diagnostic yield, diagnostic accuracy, performance of the AI tool, satisfaction with the tool or procedure

        
The following is the description of the condition:
<condition> {condition} </condition>

The following is the description of the primary outcome:
<primary_outcome> {primary_outcome} </primary_outcome>

Return only the primary outcome type, nothing else."""

    SYSTEM_MESSAGE = "You are a helpful assistant that classifies primary outcomes into types based on descriptions of the condition and the primary outcome, which has a small probability of containing data entry errors."