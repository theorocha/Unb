from transformers import GPT2LMHeadModel, GPT2Tokenizer

def geraTexto(fraseInicial, max_length=50):
    modelName = "gpt2"  
    tokenizer = GPT2Tokenizer.from_pretrained(modelName)
    modelo = GPT2LMHeadModel.from_pretrained(modelName)
    input_ids = tokenizer.encode(fraseInicial, return_tensors="pt")
    
    #Linha abaixo pode ter seus parâmetros alterados conforme regra necessária.
    output = modelo.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

fraseInicial = "God is good"
print(geraTexto(fraseInicial, max_length=128))