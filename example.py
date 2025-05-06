from transformers import pipeline

# Usar modelo do tipo text-generation
generator = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

# Texto inicial (prompt)
prompt = "Conte me uma história?"

# Gerar texto
generated_text = generator(prompt, temperature=0.85)

# Exibir o texto gerado
print(generated_text[0]['generated_text'])



