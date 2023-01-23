import gradio as gr
from transformers import pipeline


model_checkpoint = "shubham555/biobert-finetuned-ner"
token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")


examples = [
    ["Clustering of missense mutations in the ataxia - telangiectasia gene in a sporadic T - cell leukaemia."],
    ["Ataxia - telangiectasia ( A - T ) is a recessive multi - system disorder caused by mutations in the ATM gene at 11q22 - q23 ( ref . 3 )."],
    ["The risk of cancer , especially lymphoid neoplasias , is substantially elevated in A - T patients and has long been associated with chromosomal instability."],
    ["These clustered in the region corresponding to the kinase domain , which is highly conserved in ATM - related proteins in mouse , yeast and Drosophila."],
    ["Constitutional RB1 - gene mutations in patients with isolated unilateral retinoblastoma ."],
    ["The evidence of a significant proportion of loss - of - function mutations and a complete absence of the normal copy of ATM in the majority of mutated tumours establishes somatic inactivation of this gene in the pathogenesis of sporadic T - PLL and suggests that ATM acts as a tumour suppressor."],
]


def ner(text):
    output = token_classifier(text)
    for hmap in output:
      hmap['entity'] = hmap['entity_group']
      del hmap['entity_group']
    return {"text": text, "entities": output}    

demo = gr.Interface(ner,
             gr.Textbox(placeholder="Enter sentence here..."), 
             gr.HighlightedText(),
             examples=examples,
             allow_flagging = 'never',
             title="Named Entity Recognition for Disease Identification",
             description="The app uses BioBERT finetuned on NCBI Dataset and can be used to detect the name of diseases appearing in the given text")

demo.launch()
