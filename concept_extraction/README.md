# Readme for concept_extraction

The code and data in this directory correspond to the blog post:
[https://appliedingenuity.substack.com/  Build a Domain-Specific, Spell Correcting, Concept Extractor in Less Than One Day ](https://appliedingenuity.substack.com/p/build-a-domain-specific-spell-correcting)

The data file ./extracted_baseball.json was built using rec.sport.baseball from: http://qwone.com/~jason/20Newsgroups/ 20news-18828.tar.gz

# Important files:
./make_training_concepts.py <-- run to generate examples for fine_tuning
./extract_concepts.py <-- run to extract concepts after you have fine-tuned your model

# NOTE:
These scripts might require you to have an OpenAI API key.


To use this code, edit your ~/.env file to include:
OPENAI_API_KEY=sk-<YOUR_OPENAI_KEY>
EXTRACTION_MODEL=<YOUR_FINE_TUNED_MODEL_FOR_EXTRACTION>

you can get help with any of these scripts with --help
