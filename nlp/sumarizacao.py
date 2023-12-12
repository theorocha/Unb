from transformers import BartForConditionalGeneration, BartTokenizer

def sumarizar(textoGrande, max_length=150):
    modelo = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(modelo)
    tokenizer = BartTokenizer.from_pretrained(modelo)

    inputs = tokenizer(textoGrande, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summary

# Exemplo de uso
textoGrande = """
    The use of artificial intelligence (AI) has grown rapidly across various industries in recent years. Companies are adopting AI-based solutions to optimize processes, make more informed decisions, and drive innovation. In the healthcare sector, AI is being used to diagnose diseases, personalize treatments, and improve operational efficiency.
Furthermore, AI has played a crucial role in technological advancements such as autonomous vehicles, speech recognition, automatic translation, and virtual assistants. However, the rapid advancement of AI also raises ethical concerns, including issues related to privacy, algorithmic bias, and the impact on employment.
To address these challenges, AI researchers and professionals are working to develop ethical guidelines and standards for the responsible use of technology. It is essential to balance the benefits of AI with ethical and societal considerations to ensure that it is used ethically and sustainably.
In summary, artificial intelligence is rapidly transforming the business and technological landscape, offering significant opportunities but also challenging us to address complex ethical issues.
    """
print("Sumarização:", sumarizar(textoGrande, max_length=150))
