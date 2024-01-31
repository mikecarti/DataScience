# Overview 
https://drive.google.com/file/d/1uj5Utd0bCMFSLAFq_EJBQxpXY5PxOG8b/view?usp=sharing

# Dependencies for linux:
Use python3.10

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
spacy download en_core_web_sm
python main.py
```

If does not work:
```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git 
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
python main.py
```

# Process
### Research
Searched for good embeddings, that would fit on my 4GB GPU, 
decided to stop on intfloat/e5-base, as it is lightweight and pretty good, compared to alternatives.
- https://huggingface.co/spaces/mteb/leaderboard

### Ideas for similar phrase detection
First of all, I thought of all ideas, that could work for a text
improvement engine. It was obvious, that I need to work with word representations,
but it was not obvious, how to capture phrases. First idea was to just split
sentences by words, and analyze word by word, but single word is not able to 
capture phrase complexity, then there was another extreme: feed whole sentence into
similarity search. In latter case, we may encounter problems such as:
1) Long sentences vector representations contain too much noise with respect to business phrases
2) We may encounter several phrases, and we will not be able to suggest the exact way of replacing these phrases.


Thus the new idea was born: to find a way of splitting a sentence by its phrases. After 2 hours
of research, i decided to stop on a solution with benepar library.

### Similarity measure
```math
\begin{align}
S(a,b) = \text{cosSim}(a, b) * \Big( \frac{log(\text{wordCount(a)})}{log(\text{MaxDocWordCount})} \Big)^\alpha
\end{align}
```

For similarity measure I came up with that similarity function. I took simple cosine_similarity multiplied and normalized 
with natural logarithm of word count to fight short phrase scoring problem. $\alpha$ is a hyper-parameter,
describing how much score is supposedly affected by word count.

### Stack
I decided to use FAISS vector database, as it is convinient
to store vector representations there. For task of sentence splitting (i.e split a sentence
by phrases), I researched a lot of options, but as this field is 
not very popular (sentence segmentation) I decided to go with a good-enough library - benepar.
Model loading and inference is done through nltk and huggingface.

### Text Improvement
I decided to not stop with suggesting phrases, but also added contextualisation,
given a large enough LLM, programm will suggest not only which words to change, but
how to change them as well.

### Future Improvements
- Better Embeddings (e5 Mistral 7B)
- Better LLM (LLAMA 2)
- Tune hyper-parameters (LLM hyper-parameters, norm_scaling_factor, similarity threshold)
- Hyper Parameters may be tuned with use of collecting dataset for a classification task and Grid Search
- Find a better function for similarity measure
- Try to add business-like vectors to analyzed vectorized phrases with context-smart attention mechanism
- Fine-tune model for this specific task
- Train a model for sentence segmentation task

### Results
Specific preloaded phrases are located in the file standard_phrases.txt. Sample text for
analysis is located in text.txt. All scores were obtained with e5_base embeddings.

##### GPT-4 (Any locally deployed big enough LLM would produce similar results)
```commandline
Phrase: came to the consensus that we need to do better in terms of performance
Candidates:
[0.6674759451321894]: Optimal performance
[0.6329728259200568]: Enhance productivity

Suggested phrase: Came to the consensus that we need to achieve optimal performance.
 ----------------------------------------------------------------------------------------------------
Phrase: 's important to make good use of what we have at our disposal
Candidates:
[0.6660369794742202]: Utilise resources

Suggested phrase: It's important to utilise resources we have at our disposal.
 ----------------------------------------------------------------------------------------------------
Phrase: should aim to be more efficient and look for ways to be more creative in our daily tasks
Candidates:
[0.7021939026323047]: Enhance productivity
[0.6723416743644108]: Utilise resources
[0.6680336870337832]: Implement best practices

Suggested phrase: We should aim to utilise resources to be more efficient and look for ways to implement best practices to be more creative in our daily tasks.
 ----------------------------------------------------------------------------------------------------
Phrase: agreed that we must take time to look over our plans carefully and consider all angles before moving forward
Candidates:
[0.6467422601589206]: Exercise due diligence
[0.6336928617049571]: Prioritise tasks

Suggested phrase: Agreed that we must exercise due diligence to look over our plans carefully and consider all angles before moving forward.
 ----------------------------------------------------------------------------------------------------
```

##### Quantized open_llama_3b (less then 4GB RAM)
```commandline
Phrase: came to the consensus that we need to do better in terms of performance
Candidates:
[0.6674759451321894]: Optimal performance
[0.6329728259200568]: Enhance productivity

Suggested phrase:  We
 ----------------------------------------------------------------------------------------------------
Phrase: 's important to make good use of what we have at our disposal
Candidates:
[0.6660369794742202]: Utilise resources

Suggested phrase:  's important to make good use of what we have at our disposal.
 ----------------------------------------------------------------------------------------------------
Phrase: should aim to be more efficient and look for ways to be more creative in our daily tasks
Candidates:
[0.7021939026323047]: Enhance productivity
[0.6723416743644108]: Utilise resources
[0.6680336870337832]: Implement best practices

Suggested phrase:   should aim to be more efficient and look for ways to be more creative in our daily tasks.
 ----------------------------------------------------------------------------------------------------
Phrase: agreed that we must take time to look over our plans carefully and consider all angles before moving forward
Candidates:
[0.6467422601589206]: Exercise due diligence
[0.6336928617049571]: Prioritise tasks

Suggested phrase:  We need
 ----------------------------------------------------------------------------------------------------
```
