#!/usr/bin/env python3
"""
Test des limites de tokens pour Kibali
Vérifie que le modèle peut générer 3000 tokens
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("=" * 60)
print("TEST DES LIMITES DE TOKENS - KIBALI")
print("=" * 60)

# Chemins des modèles
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"\n📦 Chargement du modèle: {model_name}")

try:
    # Charger le tokenizer
    print("\n1️⃣ Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ Tokenizer chargé (vocab size: {len(tokenizer)})")
    
    # Charger le modèle
    print("\n2️⃣ Chargement du modèle...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device.upper()}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print(f"   ✅ Modèle chargé sur {device.upper()}")
    
    # Test 1: Génération courte (baseline)
    print("\n" + "=" * 60)
    print("TEST 1: Génération courte (500 tokens)")
    print("=" * 60)
    
    prompt1 = """Explique-moi en détail comment fonctionne la tomographie de résistivité électrique (ERT). 
Décris le principe physique, les équipements utilisés, et les applications en géophysique."""
    
    messages1 = [
        {"role": "system", "content": "Tu es un expert en géophysique."},
        {"role": "user", "content": prompt1}
    ]
    
    inputs1 = tokenizer.apply_chat_template(
        messages1,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    attention_mask1 = (inputs1 != tokenizer.pad_token_id).long().to(device)
    
    start1 = time.time()
    with torch.no_grad():
        outputs1 = model.generate(
            inputs1,
            attention_mask=attention_mask1,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    time1 = time.time() - start1
    
    response1 = tokenizer.decode(outputs1[0][inputs1.shape[1]:], skip_special_tokens=True)
    tokens1 = len(tokenizer.encode(response1))
    
    print(f"\n📊 Résultats:")
    print(f"   Temps: {time1:.2f}s")
    print(f"   Tokens générés: {tokens1}")
    print(f"   Longueur réponse: {len(response1)} caractères")
    print(f"\n📝 Début de la réponse:")
    print(f"   {response1[:300]}...")
    
    # Test 2: Génération longue (3000 tokens)
    print("\n" + "=" * 60)
    print("TEST 2: Génération longue (3000 tokens)")
    print("=" * 60)
    
    prompt2 = """Tu dois analyser en profondeur le fichier 'Projet Archange Ondimba 2.dat' qui contient des données de tomographie électrique.

Données du fichier:
- Type: Tomographie de Résistivité Électrique (ERT)
- Format: Fichier .dat avec mesures de résistivité
- Profondeur: 0 à 50 mètres
- Valeurs de résistivité: de 10 à 5000 Ohm.m
- Points de mesure: 250 valeurs
- Localisation: Gabon, projet Archange Ondimba

Exemple de valeurs:
Profondeur 0m: 45.2, 52.1, 48.9 Ohm.m
Profondeur 5m: 78.3, 82.5, 91.2 Ohm.m
Profondeur 10m: 125.4, 132.8, 140.2 Ohm.m
Profondeur 15m: 245.6, 258.9, 267.3 Ohm.m

Fais une analyse COMPLÈTE et DÉTAILLÉE incluant:
1. Interprétation géologique des valeurs
2. Identification des couches et leur nature (argile, sable, roche)
3. Potentiel hydrogéologique 
4. Recommandations pour exploration
5. Analyse statistique des valeurs
6. Comparaison avec autres sites similaires
7. Suggestions d'investigations complémentaires

Sois très détaillé et explicite chaque point."""
    
    messages2 = [
        {"role": "system", "content": "Tu es un expert en géophysique et hydrogéologie."},
        {"role": "user", "content": prompt2}
    ]
    
    inputs2 = tokenizer.apply_chat_template(
        messages2,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    attention_mask2 = (inputs2 != tokenizer.pad_token_id).long().to(device)
    
    print(f"\n🚀 Génération en cours avec max_new_tokens=3000...")
    start2 = time.time()
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast():
                outputs2 = model.generate(
                    inputs2,
                    attention_mask=attention_mask2,
                    max_new_tokens=3000,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    use_cache=True
                )
        else:
            outputs2 = model.generate(
                inputs2,
                attention_mask=attention_mask2,
                max_new_tokens=3000,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05
            )
    time2 = time.time() - start2
    
    response2 = tokenizer.decode(outputs2[0][inputs2.shape[1]:], skip_special_tokens=True)
    tokens2 = len(tokenizer.encode(response2))
    
    print(f"\n📊 Résultats:")
    print(f"   Temps: {time2:.2f}s")
    print(f"   Tokens générés: {tokens2}")
    print(f"   Longueur réponse: {len(response2)} caractères")
    print(f"   Ratio temps/token: {time2/tokens2*1000:.2f}ms/token")
    
    print(f"\n📝 Réponse COMPLÈTE:")
    print("=" * 60)
    print(response2)
    print("=" * 60)
    
    # Vérifier si la réponse est coupée
    if "..." in response2[-50:] or tokens2 < 2500:
        print("\n⚠️  ATTENTION: La réponse semble coupée!")
        print(f"   Tokens générés: {tokens2} (attendu: proche de 3000)")
    else:
        print("\n✅ SUCCESS: Réponse complète générée!")
        print(f"   {tokens2} tokens générés sur 3000 max")
    
    # Comparaison
    print("\n" + "=" * 60)
    print("COMPARAISON DES TESTS")
    print("=" * 60)
    print(f"Test 1 (500 tokens):  {tokens1} tokens en {time1:.2f}s")
    print(f"Test 2 (3000 tokens): {tokens2} tokens en {time2:.2f}s")
    print(f"Gain de longueur: +{tokens2-tokens1} tokens ({(tokens2/tokens1-1)*100:.1f}%)")
    
    print("\n✅ Tests terminés avec succès!")
    
except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
